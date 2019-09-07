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

const int Iteration_recordNum=6*2*32*16;
const int MAX_Record=1e3;      //设置的子集最大记录数量（不能保证划分后的记录数量）
const int Key_Length=6;        //记录的键值最大长度
const int File_num=1e5;       //记录数量
int key_num=0;                //未重复记录数量
const sharedMem_size=32*32*sizeof(float);



struct Data{
	int edge_num[2];
	string str;
};
vector<Data> CPU_Data;

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



int main(){
	create_IO_File(path);
	deal_IO_File(path);
	key_num=CPU_Data.size();
	
	//读取数据（键值，误差范围）
	int *temporary_num=(int *)malloc(key_num*2*sizeof(int));
	int *temporary_key=(int *)malloc(key_num*Key_Length*sizeof(int));
	int *temporary_result=(int *)malloc(key_num*sizeof(int));
	for(int i=0;i<key_num;i++){
		int u=i*2;
		int m=i*Key_Length;
		string str=CPU_Data[i].str;
		temporary_num[u]=CPU_Data[i].edge_num[0];
		temporary_num[u+1]=CPU_Data[i].edge_num[1];
		u=0;
		for(;u<Key_Length;u++){
			for(auto c:str){
				temporary_key[m+u]=c;
			}
			temporary_key[m+u]='\0';
		}
	}
	//设定将数据一次性 传递到GPU中
	
	Error_judge(cudaMalloc((void**)&GPU_edge_num,key_num*2*sizeof(int)),__LINE__);
	Error_judge(cudaMalloc((void**)&GPU_Data,key_num*Key_Length*sizeof(int)),__LINE__);
	
	Error_judge(cudaMemcpy(GPU_edge_num,temporary_num,key_num*2*sizeof(int),cudaMemcpyHostToDevice),__LINE__);
	Error_judge(cudaMemcpy(GPU_Data,temporary_key,key_num*Key_Length*sizeof(int),cudaMemcpyHostToDevice),__LINE__);
	
	
	//设定一次iteration的GPU内存大小
	
	Error_judge(cudaMalloc((void**)&GPU_run_result,Iteration_recordNum*sizeof(float)),__LINE__);
	Error_judge(cudaMalloc((void**)&GPU_run_position,Iteration_recordNum*sizeof(int)),__LINE__);
	
	//其他的GPU内存申请
	
	Error_judge(cudaMalloc((void**)&Partition_num,key_num*sizeof(int)),__LINE__);


	
	//进行划分前的参数初始化。
	int leaf_node=0;

	leaf_node=key_num%MAX_Record!=0? key_num/MAX_Record:key_num/MAX_Record+1;    //得到第二层子节点的个数
	
	float Param[Key_Length+1];                  //第一次的参数用于划分，尽量不要太小（不要为小数）
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


	
	//GPU中的一些常量内存
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
		vector<int> p;
		Node_vector.push_back(p);
	}
	int temp=0;
	for(int i=0;i<key_num;i++){        //划分成leaf_node个子数据集，每个数据集中的数据个数不确定
		temp=temporary_result[i];
		Node_vector[temp].push_back(i);
	}
	for(int i=0;i<Key_Length+1;i++){
		Param[i]=Param[i]/sum;
	}
	Error_judge(cudaMemcpyToSymbol(GPU_Param,Param,sizeof(Param)),__LINE__);

	for(int i=0;i<aa[0];i++){          //子节点进行线性逼近
		if(Node_vector[i].size()==0) continue;    //子节点为空继续
		
		int num1=Node_vector[i].size()/Iteration_recordNum;
		int num2=Node_vector[i].size()%Iteration_recordNum;
		if(num2!=0) num1++;
		
		//进行逼近
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
		Error_judge(cudaMemcpyToSymbol(Model_Param+i*(Key_Length+1),GPU_Param,(Key_Length+1)*sizeof(float)),__LINE__);
		time=0;
		
		//查看是否逼近成功，失败则为其建立B+树
		time=0;
		iteration_num=0;
		int flag=0;
		while(time<num1){
			if (time == num1 - 1 && num2 != 0) {
				iteration_num = num2;
			}
			else {
				iteration_num = Iteration_recordNum;
			}
			Error_judge(cudaMemcpy(GPU_run_position, &vector[i][0]+time*Iteration_recordNum, iteration_num* sizeof(int), cudaMemcpyHostToDevice), __LINE__);
			Forward_calcu<<<gird,block>>>(GPU_run_position,GPU_Data,GPU_edge_num，GPU_run_result,iteration_num);
			sum_average << <grid, block, sharedMem_size >> > (GPU_run_result, GPU_run_result, iteration_num); 
			sum_average << <1, 32, sharedMem_size >> > (GPU_run_result, GPU_run_result, 12);
			float a=-1;
			Error_judge(cudaMemcpy(&a,GPU_run_result,sizeof(float)),__LINE__);
			if(a!=0){
				flag=1;
				break;
			}
			time++;
		}
		
		if(flag==0) ;
		else Tree_struct(vector[i]); //构建B+树
		
		
		
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
			data.edge_num[1]=i-1;             //num[0]为左标号，num[1]为右标号
			if(flag==1) {
				CPU_Data.push_back(data);
				flag=0;
			}
			data.edge_num[0]=i;
			data.str=s;
			flag=1;
			str=s;
		}else ;
		
	}
}


void Model_Train(int iteration_num){
	int i=1000;
	while(i++>=0){
		Train_Forward<<<gird,block>>>(GPU_run_position,GPU_Data,GPU_edge_num，GPU_run_result,iteration_num);
		sum_average << <grid, block, sharedMem_size >> > (GPU_run_result, GPU_run_result, iteration_num); 
		sum_average << <1, 32, sharedMem_size >> > (GPU_run_result, GPU_run_result, 12);
		Train_backward<<<1,1>>>(GPU_run_result,iteration_num);
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
	int i=time*Num_Flag[1],p=0;
	GPU_run_position[tid]=0;
	if(tid<iteration_num){
		p=(i+tid)*2;
		GPU_run_position[tid]=i+tid;
		int m=(GPU_edge_num[p]+GPU_edge_num[p+1])/2;          //将误差的左右范围转换为中间值和偏差
		int n=(m-GPU_edge_num[p])>(GPU_edge_num[p+1]-m)?(m-GPU_edge_num[p]):(GPU_edge_num[p+1]-m);
		GPU_edge_num[p]=m;
		GPU_edge_num[p+1]=n;
	}
	
}

__global__ void Line_partition(int* GPU_Data,int *GPU_run_postion,float *GPU_run_result,int *Partition_num,int iteration_num,int time){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int offset=0,j=0;
	offset=GPU_run_position[tid]*(Num_Flag[2]-1);           //在GPU当前记录保存位置距离起始地址的偏移量
	GPU_run_result[tid]=0;
	if(tid<iteration_num){
		for(j=0;j<Num_Flag[2]-1;j++)
			GPU_run_result[tid]+=GPU_Data[offset+j]*GPU_Param[j];
		GPU_run_result[tid]+=GPU_Param[j];
		Partition_num[time*Num_Flag[1]+tid]=(int) GPU_run_result%Num_Flag[0];           //计算出所属子节点编号
	}
	
}





__global__ void Train_Forward(int *GPU_run_position,int *GPU_Data,int *GPU_edge_num，float *GPU_run_result,int iteration_num){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int j=0,offset=GPU_run_position[tid]*(Num_Flag[2]-1);           //键值，距离起始地址的偏移量
	int offset2=GPU_run_position[tid]*2;                            //范围数据，距离起始地址的偏移量
	GPU_run_result[tid]=0;
	if(tid<iteration_num){
		for(j=0;j<Num_Flag[2]-1;j++)
			GPU_run_result[tid]+=GPU_Data[offset+j]*GPU_Param[j];
		GPU_run_result[tid]+=GPU_Param[j];
		GPU_run_result[tid]=2*(GPU_run_result[tid]-GPU_edge_num[offset2]); //偏离值的2倍，作为偏导的一部分保存
		
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

__global__ void Train_backward(float *GPU_run_result,int iteration_num){
	float i=GPU_run_result[0]/iteration_num;
	int j=0;
	for（;j<Num_Flag[2]-1;j++）{
		GPU_Param[j]=GPU_Param[j]*(1-i*Num_Flag[3]);  //Flag[3]保存着一个常量的学习率
	}
	GPU_Param[j]=GPU_Param[j]*(1-Num_Flag[3]);
}

__global__ void Forward_calcu(int *GPU_run_position,int *GPU_Data,int *GPU_edge_num，float *GPU_run_result,int iteration_num){
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int j=0,offset=GPU_run_position[tid]*(Num_Flag[2]-1);           //键值，距离起始地址的偏移量
	int offset2=GPU_run_position[tid]*2;                            //范围数据，距离起始地址的偏移量
	GPU_run_result[tid]=0;
	if(tid<iteration_num){
		for(j=0;j<Num_Flag[2]-1;j++)                                         //Num_Flag[2]-1为键长
			GPU_run_result[tid]+=GPU_Data[offset+j]*GPU_Param[j];
		GPU_run_result[tid]+=GPU_Param[j];
		int disparity=GPU_run_result[tid]-GPU_edge_num[offset2];
		if(disparity<0) disparity=-disparity;
		if(disparity<Num_Flag[4]) GPU_run_result[tid]=0;                   //Num_Flag[4]保存着最大误差值（此值对于所有叶子节点统一）
	}
}


void Tree_struct(vector<int> data_postion){
	int num=data_postion.size();
	if(num==0) return;
	int *key=(int*) malloc(num*sizeof(int));
	int *key_param=(int*) malloc((Key_Length+1)*sizeof(int));
	for(int i=0;i<Key_Length+1;i++){
		key_param[i]=i;
	}
	memset(key,0,sizeof(key));
	for(int i=0;i<num;i++){
		int f=Key_Length*i;
		for(int j=0;j<Key_Length;j++){
			key[i]+=temporary_key[f+j]*key_param[j];
		}
	}
	
	free(key);
	free(key_param);
	
	
	
}



