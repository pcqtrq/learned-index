#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include<iostream>
#include<fstream>
#include<string>
#include<random>
#include<stdlib.h>
#include<vector>
#include "math.h"

using namespace std;

// ios::sync_with_stdio(false);


string path[2]={"C:\\data\\read.txt","C:\\data\\search.txt"};
int MAX_Error=100;


dim3 gird(6,2),block(32,16);

const int block_num=6*2;         //线程块的大小
const int Iteration_recordNum=6*2*32*16;
const int MAX_Record=1e3;      //设置的子集最大记录数量（不能保证划分后的记录数量）
const int Key_Length=6;        //记录的键值最大长度
const int File_num=1e5;       //记录数量
int key_num=0;                //未重复记录数量
const sharedMem_size=32*32*sizeof(float);

vector<string> CPU_Data;

const int K=6 ;                //限制的最大子数据集的记录数量，第二阶段划分数据集个数，
                               //线性逼近函数的参数个数，逼近过程中的学习率

//GPU中的常量内存
__constant__ float GPU_Param[Key_Length+1];
__constant__ float Num_Flag[K];

//GPU_edge_num和GPU_Data保存着GPU中所有的元数据

__device__ int *GPU_edge_num=NULL;
__device__ int *GPU_Data=NULL;

//GPU_run_position以及GPU_run_result是GPU中一次Iteration的最大计算数量，其size被设置为常量

__device__ float *GPU_run_result=NULL;
__device__ int *GPU_run_position=NULL;
__device__ float *Model_Param=NULL;


//Paration_num保存GPU中所有数据的进行顶层划分后的子节点编号计算结果
__device__ int *Partition_num=NULL;

/*反馈神经网络需要注意两点：
（1）神经网络分为三层：输入层，规划层，输出层
（2）多样本的神经网络反向传递可以分为三部分：一是前向传递中每层中每个神经元计算数据的平均值以及激活函数的导数平均值，一是损失函数，一是损失函数对于神经网络中进行计算的每个神经元的导数或偏导

*/
const int NN_Str[4]={3,Key_Length,12,1};  //神经网络结构：3层｛Key_Length，12，1｝,其中第一层为输入层
const int SIZE_nnP=(Key_Length+1)*12+(12+1)*1;
const int SIZE_Ave=(Key_Length+12+1);
const int SIZE_Def=(1+Key_Length+12);

__constant__ float GPU_NN_Str[4];
__constant__ float NN_Param[SIZE_nnP];        //模型参数
__constant__ float Average_Data[SIZE_Ave];   //迭代中的均值
__constant__ float Deflection[SIZE_Def];           //迭代中的偏导
__constant__ float Active[12];          //激活函数的导数

__device__ float *layer1_result,*layer1_derivative,*temp;  //第一层的计算结果，第一层的激活函数的导数，以及一个共享性的公用内存空间


//过滤网
//这里打算使用vector对过滤网进行记录,（可能会更换过滤网的组织格式以便进行简化）


vector<string> Grid_Unit_S;


int main(){
	create_IO_File(path);
	deal_IO_File(path);
	key_num=CPU_Data.size();
	
	//读取数据（键值，误差范围）
	int *temporary_key=(int *)malloc(key_num*Key_Length*sizeof(int));
	int *temporary_result=(int *)malloc(key_num*sizeof(int));           //用于存储数据集划分结果
	for(int i=0;i<key_num;i++){
		int u=i*2;
		int m=i*Key_Length;
		string str=CPU_Data[i];
		u=0;
		for(;u<Key_Length;u++){
			for(auto c:str){
				temporary_key[m+u]=c;
			}
			temporary_key[m+u]='\0';
		}
	}
	//设定将数据一次性 传递到GPU中
	
	Error_judge(cudaMalloc((void**)&GPU_Data,key_num*Key_Length*sizeof(int)),__LINE__);
	Error_judge(cudaMemcpy(GPU_Data,temporary_key,key_num*Key_Length*sizeof(int),cudaMemcpyHostToDevice),__LINE__);
	
	
	//设定一次iteration的GPU内存大小
	
	Error_judge(cudaMalloc((void**)&GPU_run_result,Iteration_recordNum*sizeof(float)),__LINE__);
	Error_judge(cudaMalloc((void**)&GPU_run_position,Iteration_recordNum*sizeof(int)),__LINE__);
	
	//其他的GPU内存申请
	
	Error_judge(cudaMalloc((void**)&Partition_num,key_num*sizeof(int)),__LINE__);


	
	//进行划分前的参数初始化。
	int leaf_node=0;

	leaf_node=key_num%MAX_Record!=0? key_num/MAX_Record:key_num/MAX_Record+1;    //得到第二层子节点的个数
	
	float Param[Key_Length+1];                  //第一次的参数用于划分，尽量不要太小（不要小于1）
	int sum=0;                                  //用于初始化后面的线性逼近参数
	for(int i=0;i<Key_Length+1;i++){
		int y=random()%100;
		Param[i]=1.0*(y+10)/y;
		sum+=Param[i];
	}
	//模型参数保存内存空间申请
	Error_judge(cudaMalloc((void**)&Model_Param,(1+leaf_node)*(Key_Length+1)*sizeof(float)),__LINE__);
	Error_judge(cudaMemcpyToSymbol(GPU_Param,Param,sizeof(Param)),__LINE__);
	Error_judge(cudaMemcpyToSymbol(Model_Param，GPU_Param,(Key_Length+1)*sizeof(float)),__LINE__);


	
	//GPU中的一些常量内存初始化
	float aa[K]={leaf_node,Iteration_recordNum,Key_Length+1,0.001,MAX_Error,0};
	Error_judge(cudaMemcpyToSymbol(Num_Flag,aa,sizeof(aa)),__LINE__);

	
	//进行数据集划分
	int num1=key_num/Iteration_recordNum;
	int num2=key_num%Iteration_recordNum;
	if(num2!=0) num1++;

	int time=0;
	int iteration_num=0;
	while(time<num1){
		if (time == num1 - 1 && num2 != 0) {
			iteration_num = num2;
		}
		else {
			iteration_num = Iteration_recordNum;
		}
		Init_position<<<grid,block>>>( time,iteration_num,GPU_run_position);
		Line_partition<<<gird,block>>>(GPU_run_position,GPU_run_result,Partition_num,iteration_num,time);
		time++;
	}
	
	Error_judge(cudaMemcpy(temporary_result,Partition_num,sizeof(temporary_result),cudaMemcpyDeviceToHost),__LINE__);
	vector<vector<int>> Node_vector;
	for (int i = 0; i < aa[0]; i++){   
		vector<int> p;                  //每个叶子节点使用一个vector<int>保存相关记录的编号
		Node_vector.push_back(p);    
	}
	int temp=0;
	for(int i=0;i<key_num;i++){        //划分成leaf_node个子数据集，每个数据集中的数据个数不确定
		temp=temporary_result[i];
		Node_vector[temp].push_back(i);
	}
	for(int i=0;i<Key_Length+1;i++){
		Param[i]=Param[i]/sum;         //sum为所有参数的和，现在所有参数和为1
	}
	Error_judge(cudaMemcpyToSymbol(GPU_Param,Param,sizeof(Param)),__LINE__);

	for(int i=0;i<aa[0];i++){          //aa[0]为叶子节点数，开始对每个叶子的数据进行训练
		if(Node_vector[i].size()==0) continue;    //子节点为空，继续
		
		//与前面划分类似，计算出该子数据的迭代次数
		int num1=Node_vector[i].size()/Iteration_recordNum;
		int num2=Node_vector[i].size()%Iteration_recordNum;
		if(num2!=0) num1++;
		
		//对叶子i的数据进行概率逼近
		int time=0;
		int iteration_num=0;
		while(time<num1){
			if (time == num1 - 1 && num2 != 0) {
				iteration_num = num2;
			}
			else {
				iteration_num = Iteration_recordNum;
			}
			
			Error_judge(cudaMemcpy(GPU_run_position, &vector[i][0]+time*Iteration_recordNum, iteration_num* sizeof(int), cudaMemcpyHostToDevice), __LINE__);
			Model_Train(iteration_num);
			time++;
		}
		Error_judge(cudaMemcpyToSymbol(GPU_Param+sizeof(Param)+i*SIZE_nnP,Param,sizeof(Param)),__LINE__);

		
		//标记不合格元素
		time=0;
		iteration_num=0;
		while(time<num1){
			if (time == num1 - 1 && num2 != 0) {
				iteration_num = num2;
			}
			else {
				iteration_num = Iteration_recordNum;
			}
			
			Error_judge(cudaMemcpy(GPU_run_position, &vector[i][0]+time*Iteration_recordNum, iteration_num* sizeof(int), cudaMemcpyHostToDevice), __LINE__);
			gird_interception(iteration_num);
			
			//下行代码执行后，会将该vector中所有已满足要求的先关键值标志设置为-1
			Error_judge(cudaMemcpy(&vector[i][0]+time*Iteration_recordNum,GPU_run_position,  iteration_num* sizeof(int), cudaMemcpyDeviceToHost), __LINE__);

			time++;
		}
		
		//建立假阴性过滤网
		for(int j=0;j<vector[i].size();j++)
			if(vector[i][j]!=-1){
				int position=vector[i][j]*Key_Length;
				string s="";
				for(int m=0;m<Key_Length;m++){
					if(temporary_key[position+m]!='\0') 
						s+=temporary_key[position+m];
					else break;
				}
				Grid_Unit_S.push_back(s);	
			}
		
		
	}
	
	

	free(temporary_num);
	free(temporary_key);
	free(temporary_result);
	cudaFree(GPU_edge_num);
	cudaFree(GPU_Data);
	cudaFree(GPU_run_position);
	cudaFree(GPU_run_result);
	cudaFree(Partition_num);
	
	



}

void create_IO_File(string path[2]){
	int j=0,y=0;
	ofstream fp(path[0]);
	for(int i=0;i<File_num;i++){
		j=random()%20;
		y=i;
		for(int n=0;n<j&&i<File_num;n++,i++){
			fp<<y<<"\n";
		}
	}
}

void deal_IO_File(string path[2]){
	ifstream inf(path[0]);
	
	if(inf.fail()){
		cerr<<path[0]<<" cannot open \n";
	}
	string str="";
	int i=0,flag=0;
	struct Data data;
	while (getline(inf, s))
	{
		i++;
		if(str!=s){
			CPU_Data.push_back(s);
			flag=0;
			str=s;
		}else ;
		
	}
}


void Model_Train(int iteration_num){
	int A[SIZE_nnP];
	for(int i=0;i<SIZE_nnP;i++){
		A[i]=(random()%100)*0.01;
	}
	//传递一些常量
	Error_judge(cudaMemcpyToSymbol(GPU_NN_Str,NN_Str,sizeof(NN_Str)),__LINE__);
	
	//传递模型参数
	Error_judge(cudaMemcpyToSymbol(NN_Param,A,sizeof(A)),__LINE__);
	
	//申请模型的内存空间
	Error_judge(cudaMalloc((void**)&layer1_result,key_num*Key_Length*sizeof(int)),__LINE__);
	Error_judge(cudaMalloc((void**)&layer1_derivative,key_num*GPU_NN_Str[2]*sizeof(int)),__LINE__);
	
	Error_judge(cudaMalloc((void**)&temp,block_num*sizeof(float)),__LINE__);
	
	
	int i=1000;           //训练次数
	while(i++>=0){
		One_Train(iteration_num);
		
		
	}
}

void Error_judge(cudaError Error, int line) {                         //用于判断数据传送是否出错
	if (Error != cudaSuccess) {
		cout << "Error !!        " << line << endl;
		exit(0);
	}
}



__global__ void Init_position(int time,int iteration_num,int *GPU_run_position){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int i=time*Num_Flag[1];       //Num_flag[1]保存每次迭代的记录数量，此时所有记录的位置编号都是连续的
	GPU_run_position[tid]=0;
	if(tid<iteration_num){
		GPU_run_position[tid]=i+tid;
	}
}

__global__ void Line_partition(int* GPU_Data,int *GPU_run_postion,float *GPU_run_result,int *Partition_num,int iteration_num,int time){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int offset=0,j=0;
	offset=GPU_run_position[tid]*(Num_Flag[2]-1);           //Num_Flag[2]保存着键长+1，即线性表达式参数，减1则为键长
	GPU_run_result[tid]=0;
	if(tid<iteration_num){
		for(j=0;j<Num_Flag[2]-1;j++)
			GPU_run_result[tid]+=GPU_Data[offset+j]*GPU_Param[j];
		GPU_run_result[tid]+=GPU_Param[j];                                              //保存着某个迭代中某一记录所示子节点编号
		Partition_num[time*Num_Flag[1]+tid]=(int) GPU_run_result%Num_Flag[0];           //Partition_num保存着记录的所有记录子节点编号
	}
	
}


__global__ void sum_average(float *GPU_run_result, float *sum_result, int interation_num){
	int threadId = threadIdx.x + threadIdx.y*blockDim.x;
	int blockId = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = threadId + (blockDim.x*blockDim.y)*blockId;
	extern __shared__ float sdata[];
	float x = 0.0;
	if (tid < length)
	{
		x = GPU_run_result[tid];
	}
	sdata[threadIdx.x] = x;
	__syncthreads();           //等待所有线程把自己负责的元素载入到共享内存

	for (int offset = blockDim.x*blockDim.y / 2;offset > 0;offset >>= 1)     //offset >>= 1等价于offset/2,现在进行的是线程块内的计算
	{
		if (threadId < offset)//控制只有某些线程才进行操作。
		{
			sdata[threadId] += sdata[threadId + offset];
		}
		__syncthreads();
	}
	if (threadId == 0)
	{
		sum_result[blockId] = sdata[0];
	}
}


void One_Train(int iteration_num){

	//前向传递
	one_layer<<<grid,block>>>(GPU_run_position,GPU_Data,layer1_result,NN_Param,iteration_num,layer1_derivative);
	//two_layer<<<grid,block>>>(layer1_result,layer2_result,NN_Param,iteration_num);
	result_probability<<<grid,block>>>(layer1_result,GPU_run_result,NN_Param,iteration_num);
	
	//计算均方差下，输出层神经元的导数
	Deviance<<<grid,block>>>(GPU_run_result,iteration_num);
	
	//Average_Data[]中保存的偏导值，地址顺序为：输出->隐藏层->输入
	
	//计算均方差导数的平均值
	sum_average<<<grid,block, sharedMem_size>>>(GPU_run_result,temp,iteration_num);
	sum_average<<<grid,block, sharedMem_size>>>(temp,temp,block_num);
	Error_judge(cudaMemcpyToSymbol(Average_Data,temp,sizeof(float)),__LINE__);
	
	//计算隐藏层第一层每个神经元的输出平均值
	for(int i=0;i<NN_Str[2];i++){
		sum_HidenLayer<<<grid,block，sharedMem_size>>>(layer1_result+i,temp,iteration_num);
		sum_average<<<grid,block, sharedMem_size>>>(temp,temp,block_num);
		Error_judge(cudaMemcpyToSymbol(Average_Data+NN_Str[3]+i,temp,sizeof(float)),__LINE__);
	}
	
	//计算隐藏层第一层的激活函数的平均值
	for(int i=0;i<NN_Str[2];i++){
		sum_HidenLayer<<<grid,block，sharedMem_size>>>(layer1_derivative+i,temp,iteration_num);
		sum_average<<<grid,block, sharedMem_size>>>(temp,temp,block_num);
		Error_judge(cudaMemcpyToSymbol(Active+i,temp,sizeof(float)),__LINE__);
	}
	
	
	//计算输入层每个神经元的平均输出值
	for(int i=0;i<NN_Str[1];i++){
		sum_InputLayer<<<grid,block，sharedMem_size>>>(GPU_run_position,GPU_Data,temp,iteration_num,i);
		sum_average<<<grid,block, sharedMem_size>>>(temp,temp,block_num);
		Error_judge(cudaMemcpyToSymbol(Average_Data+NN_Str[1]+NN_Str[2]+i,temp,sizeof(float)),__LINE__);
	}
	
	//反向传递
	Updata_Layer<<<1,32>>>(i,iteration_num);

}

__global__ void one_layer(int *GPU_run_position,int *GPU_Data，float *layer1_result,float *NN_Param,int iteration_num,float *layer1_derivative){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int offset2=tid*GPU_NN_Str[2];
	int m=0,j=0,n=0;              //n为参数起始地址偏移量，m为单个神经元计算的偏移量
	if(tid<iteration_num){
		int u=GPU_run_position[tid]*GPU_NN_Str[1];
		for(int i=0;i<GPU_NN_Str[2];i++){
			layer1_result[offset2+i]=0;
			m=i*(GPU_NN_Str[1]+1);
			for(j=0;j<GPU_NN_Str[1];j++){
				layer1_result[offset2+i]+=GPU_Data[u+j]*NN_Param[n+m+j];
			}
			layer1_result[offset2+i]+=GPU_Data[u+j]*NN_Param[n+m+j];

			//设置激活函数为sigmod函数
			layer1_result[offset2+i]=1/(1+exp(layer1_result[offset2+i]))；
			
			//sigmod激活函数的导数
			devrivative[offset2+i]=layer1_result[offset2+i]*layer1_result[offset2+i];
		}
	}

	
	
}


__global__ void result_probability(float *layer1_result,float *GPU_run_result,float *NN_Param,int iteration_num){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int m=0,j=0;
	
	//参数的起始偏移位置,计算结果的起始偏移位置，计算数据的起始偏移位置
	int n=(GPU_NN_Str[1]+1)*GPU_NN_Str[2],offset2=tid*GPU_NN_Str[3],offset1=tid*GPU_NN_Str[2]; 

	if(tid<iteration_num){
		for(int i=0;i<GPU_NN_Str[3];i++){
			m=i*(GPU_NN_Str[2]+1);
			for(j=0;j<GPU_NN_Str[2];j++){
				GPU_run_result[offset2+i]+=layer1_result[offset1+j]*NN_Param[n+m+j];
			}
			GPU_run_result[offset2+i]+=GPU_Data[u+j]*NN_Param[n+m+j];
		}
	}
	
}


__global__ void Deviance(float *GPU_run_result,int iteration_num){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	
	if(tid<iteration_num){
		GPU_run_result[tid]=2*(GPU_run_result[tid]-1);	
	}
}


__global__ void sum_HidenLayer(float *input, float *sum_result, int interation_num){
	int threadId = threadIdx.x + threadIdx.y*blockDim.x;
	int blockId = blockIdx.x + blockIdx.y*gridDim.x;
	
	 //GPU_NN_Str[2]为隐藏层第一层的神经元个数，即每个线程对应记录所占据的数据量
	int tid = (threadId + (blockDim.x*blockDim.y)*blockId)*GPU_NN_Str[2]; 
	
	extern __shared__ float sdata[];
	float x = 0.0;
	if (tid < iteration_num)
	{
		x = input[tid];      //input在传入时已经进行了【键内】的地址偏移计算
	}
	sdata[threadIdx.x] = x;
	__syncthreads();           //等待所有线程把自己负责的元素载入到共享内存

	for (int offset = blockDim.x*blockDim.y / 2;offset > 0;offset >>= 1)     //offset >>= 1等价于offset/2,现在进行的是线程块内的计算
	{
		if (threadId < offset)//控制只有某些线程才进行操作。
		{
			sdata[threadId] += sdata[threadId + offset];
		}
		__syncthreads();
	}
	if (threadId == 0)
	{
		sum_result[blockId] = sdata[0];
	}
}


__global__ void sum_InputLayer(int *GPU_run_position,int *GPU_Data,float *temp,iteration_num,i){
	int threadId = threadIdx.x + threadIdx.y*blockDim.x;
	int blockId = blockIdx.x + blockIdx.y*gridDim.x;
	
	 //tid是本线程的编号，也是所对应的键的编号，内存中每个键（键长Key_Length）顺次放置
	int tid = threadId + (blockDim.x*blockDim.y)*blockId; 

	//GPU_NN_Str[1]输入层的神经元个数，即每个线程对应记录所占据的数据量
	
	tid=GPU_run_position[tid]*GPU_NN_Str[1]+i;        //键偏移+键内偏移
	
	
	
	extern __shared__ float sdata[];
	float x = 0.0;
	if (tid < iteration_num)
	{
		x = GPU_Data[tid];     
	}
	sdata[threadIdx.x] = x;
	__syncthreads();           //等待所有线程把自己负责的元素载入到共享内存

	for (int offset = blockDim.x*blockDim.y / 2;offset > 0;offset >>= 1)     //offset >>= 1等价于offset/2,现在进行的是线程块内的计算
	{
		if (threadId < offset)//控制只有某些线程才进行操作。
		{
			sdata[threadId] += sdata[threadId + offset];
		}
		__syncthreads();
	}
	if (threadId == 0)
	{
		temp[blockId] = sdata[0];
	}
	
	
	
}


__global__ void Updata_Layer(int i,int iteration_num){
	int tid=threadIdx.x;
	if(tid==0){
		int i=0,flag=1,offset1=0,offset2=0;
		Average_Data[0]=Average_Data[0]/iteration_num;         //这里已经把输出转化为损失函数对输出神经元的导数
		Deflection[0]=Average_Data[0];                       //损失函数对输出神经元的导数
		
		offset1=(GPU_NN_Param[1]+1)*GPU_NN_Param[2];       //输出层参数的起始位置
		offset2=0;
		
		for(i=0;i<GPU_NN_Str[2];i++){  
			Deflection[flag]=0;		
			Average_Data[flag]=Average_Data[flag]/iteration_num;
			Deflection[offset2+flag]=Deflection[0]*NN_Param[offset1+i];        //损失函数对隐藏层神经元的偏导
			flag++;
		}
		
		//参数更新有这些值参与计算：参数本身，与参数进行计算的输入值（平均值），激活函数的导数，学习率
		
		
		
		//输出层参数更新,只有一个神经元，下面的计算可以看做是一个for(i=0;i<1;i++)的内部循环
		for(i=0;i<GPU_NN_Str[2];i++){
			NN_Param[offset1+i]=NN_Param[offset1+i]*(1-Deflection[offset3]*Average_Data[offset2+i]*Num_Flag[3]);
		}
		NN_Param[offset1+i]=NN_Param[offset1+i]*(1-Deflection[offset3]*Num_Flag[3]);

		
		
		//隐藏层参数更新
		for(int j=0;j<GPU_NN_Str[2];j++){        //GPU_NN_Str[2] 个神经元
			offset1=(GPU_NN_Param[1]+1)*j;           //输出层参数的起始位置
			offset2=GPU_NN_Str[3]+GPU_NN_Str[2];                                 //平均值的起始位置
			offset3=1;                                             //神经元导数的位置
			
			//对编号为j的神经元更新参数
			for(i=0;i<GPU_NN_Str[1];i++){
				NN_Param[offset1+i]=NN_Param[offset1+i]*(1-Active[i]*Deflection[1+j]*Average_Data[offset2+i]*Num_Flag[3]);
			}	
			NN_Param[offset1+i]=NN_Param[offset1+i]*(1-Active[i]*Deflection[1+j]*Num_Flag[3]);	
		}
		
	}
	
}


__global__ void gird_interception(int interation_num){
	one_layer<<<grid,block>>>(GPU_run_position,GPU_Data,layer1_result,NN_Param,iteration_num,layer1_derivative);
	//two_layer<<<grid,block>>>(layer1_result,layer2_result,NN_Param,iteration_num);
	result_probability<<<grid,block>>>(layer1_result,GPU_run_result,NN_Param,iteration_num);
	//对
	set_Flag<<<grid,block>>>(layer1_result,GPU_run_position,Iteration_recordNum);           //对于通过训练目标的键值的位置信息设置为0
	
}


// __global__ void Forward_calcu(int *GPU_run_position,int *GPU_Data,int *GPU_edge_num，float *GPU_run_result,int iteration_num){
	// int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	// int j=0,offset=GPU_run_position[tid]*(Num_Flag[2]-1);           //键值，距离起始地址的偏移量
	// int offset2=GPU_run_position[tid]*2;                            //范围数据，距离起始地址的偏移量
	// GPU_run_result[tid]=0;
	// if(tid<iteration_num){
	
		
	// }
// }


__global__ void	set_Flag(float *layer1_result,float *GPU_run_position,int Iteration_recordNum){           //对于通过训练目标的键值的位置信息设置为0
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	if(tid<iteration_num){
		if(layer1_result[tid]-1<0.2||layer1_result[tid]-1>-0.2) GPU_run_position[tid]=-1;
	}
}



