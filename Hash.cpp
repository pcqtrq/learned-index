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

#include<time.h>
using namespace std;



string path = "C:\\data\\file.txt";                                    //数据集的位置，为一个txt文件，其中每行代表一个记录
string bat_path = "C:\\data\\bat_file.txt";                            //批任务的位置，为一个txt文件，其中每行代表一个记录
//const int layer = 3;                                                 //神经网络为三层,输出层为1个神经元


const  static int record_num = 100;                                    //数据集中记录的个数
const static int bat_num = 10;                                        //检索集的内容
int oppo = 6 * 2 * 32 * 16;                                                   //每次批训练的默认样本数量
int mini_batch_size = oppo;                                             //批任务运行时的实际样本数量,初始值为oppo
const int Stage_Two = 50;
int ML_size = 1 + Stage_Two;                                           //两个阶段，每个阶段的模型均默认为神经网络，全部数据在顶层训练进行一次训练，
const static int first_neru = 16, second_neru = 16;
const static int layer_neru[2] = { first_neru,second_neru };           //第一个隐藏层与第两个隐藏层的神经元个数
dim3 grid(6, 2);                                 //标准下的批任务grid，block，共6144个线程
dim3 block(32, 16);
int ggg_size = 32 * 32 * sizeof(float);                                //动态使用共享内存时每个块内的可用最大共享内存空间
int lll_size = 32 * sizeof(float);
//倘若训练不达标，则划分为Stage_Two个区段，重新训练Stage_two个模型。
vector <int>  stageTwo[Stage_Two];                                     //划分的子数据集
float *CPU = NULL;                                                  //保存着从文件中读取出来的数据的数字化结果
char *CPU_MetaData = NULL;                                          //保存着从文件中读取出来的字符串
int num3[Stage_Two];
float *CPU_Test = NULL;
float variancep = 0.0;

const int key_length_max = 6;                                          //检索关键字的长度，【即】输入层的神经元个数,代表最大记录为e^6次方
int C[7] = { key_length_max * 2,                                      //分别为BN，layer1,layer2,layer3,b1,b2,b3的末位置
				key_length_max * 2 + key_length_max * layer_neru[0],
				key_length_max * 2 + key_length_max * layer_neru[0] + layer_neru[0] * layer_neru[1],
				key_length_max * 2 + key_length_max * layer_neru[0] + layer_neru[0] * layer_neru[1] + 1 * layer_neru[1],
				key_length_max * 2 + key_length_max * layer_neru[0] + layer_neru[0] * layer_neru[1] + 1 * layer_neru[1] + layer_neru[0],
				key_length_max * 2 + key_length_max * layer_neru[0] + layer_neru[0] * layer_neru[1] + 1 * layer_neru[1] + layer_neru[0] + layer_neru[1],
				key_length_max * 2 + key_length_max * layer_neru[0] + layer_neru[0] * layer_neru[1] + 1 * layer_neru[1] + layer_neru[0] + layer_neru[1] + 1 };

const static int MAX = C[6] + oppo * +record_num + bat_num;          //MAX取最大值,被用于多种参数的传递

struct Index_List {                   //索引结构体
	string key;
	int line;
	struct Index_List *next;

	Index_List() {
		this->line = -1;
		this->next = NULL;
	}
};
Index_List  *p[Stage_Two];                                             //索引头指针

float learning_rate = 0.00001;


float B[2] = { 0,learning_rate };
int A[4] = { key_length_max ,layer_neru[0],layer_neru[1],record_num };                          //需要传递到GPU的常量内存中的数据

int num1 = 0, num2 = 0;                                             //num1表示训练批次，num2表示最后一个批次训练数据不足oppo时的训练量

__constant__ int constant_parament[4];                               //保存一些常量
__constant__ float parament[2];                                      //parament[0]保存着学习率
__constant__ int parament_offset[7];                                 //是CPU中C[]在GPU的对应数据
__constant__ int second_model_size[2] = { Stage_Two,0 };             //second_model_size[0]保存着第二阶段划分的数据集的个数

__device__ float *train_test;                                    //train_test[0]保存着对模型训练效果的评估
__device__ float *variance;                                     //variance[0]保存着每个样本与预期值的偏差的平均值
__device__ float *A2;                                 //用于保存隐藏层第二层每个神经元的平均计算结果，是layer2_result的平均值
__device__ float *A1;                                  //用于保存隐藏层第一层每个神经元的平均计算结果，是layer1_result的平均值
__device__ float *A0;                              //用于保存BN层每个神经元的平均计算结果，是MetaData_Train的平均值
__device__ float *BN_o;                            //用于保存输入层每个神经元的平均计算结果（即输入层数据本身）的平均值
__device__ int *L2 = NULL;                                   //用于保存隐藏层第二层的计算结果经过ReLU激活函数后相对于原数据的导数的平均值
__device__ int *L1 = NULL;                                    //用于保存隐藏层第一层的计算结果经过ReLU激活函数后相对于原数据的导数的平均值

__device__ float *MetaData = NULL;                                 //GPU_MetaData用于存放训练的总体样本数据                              
__device__ int *MetaData_position = NULL;						   //MetaData用于指明MetaData_Train的数据样本在GPU_meataData中的位置
__device__ float *MetaData_Train = NULL;                           //MetaData_Train 存放着批训练的实际数据，
																   //经过BN层计算后，由于数据量不变，其数据依旧保存在MetaData_Train中
__device__ float *GPU_parament = NULL;                             //保存训练中的参数
__device__ float *Model_Parament = NULL;                           //保存着所有模型的参数
__device__ float *layer1_result = NULL, *layer2_result = NULL, *layer3_result = NULL;   //每层网络的输出值
__device__ float *GPU = NULL;                                                           //这是个万精油   
__device__ float *Param = NULL;                                                           //迭代参数的中间存储
__device__ float *result = NULL;                                                          //这也是个万精油



__host__ void deleteList(Index_List *op, int i);                                                  //删除索引结构
__host__ void Init(float *f, int sum);                                                         //初始化权重以及常量
__host__ void create(int record_num, int bat_num);                                             // 创建一个数据集，一个批任务
__host__ void deal(float *CPU, char *CPU_MetaDate, string s, int i);                                              //读取数据集中的关键字，这里假定以第一列的值作为检索对象，故而直接从第一个字符开始记录
__host__ void Stage_One_Train();
__host__ void Stage_Two_Train();
__host__ void Model_Train(int mini_batch_size, int flag, int i, int num);
__host__ void Model_forward(int mini_batch_size);
__host__ void Model_reverse(int mini_batch_size, int flag, int i, int num);
__host__ void bat_search();

__global__ void Model_Judge(int *MetaData_position, float *layer3_result, float *result, int length);
__global__ void ML_Index_Variance(float *layer3_result, int offset, int length, int num);
__global__ void Deal_index(float *layer3_result, float *result, int offset, int length, int f_num);
__global__ void RESULT(float *layer3_result, float *result, int offset, int length);
__global__ void ML_Param_Update_Before(float *GPU_parament, float *Param, int mini_batch_size, int *L1, int *L2, float *variance);
__global__ void ML_Param_Update(float *Param, float *GPU_parament, int mini_batch_size, float *A2, float *A1, float *A0, float *BN_l, float *variance);
__global__ void ML_Init_Data(int *MetaData_position, int mini_batch_size, int offset);
__global__ void BN(float *MetaData, float *MetaData_Train, int *MetaData_position, float *GPU_parament, int length);
__global__ void ML_One(float *MetaData_Train, float *GPU_parament, float *layer1_result, int length, int *L1);
__global__ void	ML_Two(float *layer1_result, float *GPU_parament, float *layer2_result, int length, int *L2);
__global__ void	ML_Three(float *layer2_result, float *GPU_parament, float *layer3_result, int length);
__global__ void ML_Variance(float *layer3_result, int *MetaData_position, int length);
__global__ void Sum_Result_BN(float *input, int *position, float *sum_result, int length, int i);
__global__ void Sum_Result(float *input, float *sum_result, int length, int i, int offset);


void Error_judge(cudaError Error, int line) {                         //用于判断数据传送是否出错
	if (Error != cudaSuccess) {
		cout << "Error !!        " << line << endl;
		exit(0);
	}
}

int main()
{
	//创建数据集
	create(record_num, bat_num);
	
	

	//从数据集中读取所有键值链表，以固定长度将检索键值按float类型读取到内存中，地址为CPU
	CPU = (float *)malloc(record_num * key_length_max * sizeof(float));
	CPU_MetaData = (char *)malloc(record_num * key_length_max * sizeof(char));
	CPU_Test = (float *)malloc(MAX * sizeof(float));
	if (CPU == NULL || CPU_MetaData == NULL || CPU_Test == NULL) {
		cout << "ORZ:申请内存失败" << endl;
		exit(0);
	}

	char str[key_length_max];
	int i = 0;

	ifstream in(path);
	if (!in) cout << "文件打开异常" << endl;
	string s;
	while (getline(in, s))
	{
		
		deal(CPU, CPU_MetaData, s, i);
		
		i = i + key_length_max;
	}
	in.close();
	cout << "文件处理结束！" << endl;
	
	cudaSetDevice(0);

	/***常量初始化***/
	Error_judge(cudaMemcpyToSymbol(constant_parament, A, 4 * sizeof(int)), __LINE__);
	Error_judge(cudaMemcpyToSymbol(parament, B, 2 * sizeof(float)), __LINE__);
	Error_judge(cudaMemcpyToSymbol(parament_offset, C, 7 * sizeof(int)), __LINE__);

	Error_judge(cudaMalloc((void**)&variance, sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&A2, second_neru * sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&A1, first_neru * sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&A0, key_length_max * sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&BN_o, key_length_max * sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&L2, second_neru * sizeof(int)), __LINE__);
	Error_judge(cudaMalloc((void**)&L1, first_neru * sizeof(int)), __LINE__);
	Error_judge(cudaMalloc((void**)&train_test, sizeof(float)), __LINE__);

	Error_judge(cudaMalloc((void**)&MetaData, record_num*key_length_max * sizeof(float)), __LINE__);               //总体数据
	Error_judge(cudaMalloc((void**)&MetaData_Train, oppo*key_length_max * sizeof(float)), __LINE__);    //训练中的数据
	Error_judge(cudaMalloc((void**)&MetaData_position, oppo * sizeof(int)), __LINE__);                   //训练数据的位置
	Error_judge(cudaMalloc((void**)&GPU_parament, C[6] * sizeof(float)), __LINE__);                                //训练中的参数
	Error_judge(cudaMalloc((void**)&Model_Parament, ML_size*C[6] * sizeof(float)), __LINE__);    //保存每个模型训练的参数，以及两个引导参数（暂时不用）。
	Error_judge(cudaMalloc((void**)&layer1_result, oppo *layer_neru[0] * sizeof(float)), __LINE__);                //第一层每个神经元计算结果
	Error_judge(cudaMalloc((void**)&layer2_result, oppo*layer_neru[1] * sizeof(float)), __LINE__);                //第二层每个神经元计算结果
	Error_judge(cudaMalloc((void**)&layer3_result, oppo * 1 * sizeof(float)), __LINE__);                              //第三层每个神经元计算结果
	Error_judge(cudaMalloc((void**)&GPU, MAX * sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&Param, (A[0] + A[1] + A[2]) * sizeof(float)), __LINE__);             //保存前向传递的某些中间参数
	Error_judge(cudaMalloc((void**)&result, record_num * sizeof(float)), __LINE__);                     //最终结果，用于划分子数据集以及其他

	/******* 初始化GPU中的数据******/
	Error_judge(cudaMemcpy(MetaData, CPU, record_num*key_length_max * sizeof(float), cudaMemcpyHostToDevice), __LINE__);         //将全部数据由CPU传递给GPU
	Error_judge(cudaMemcpy(CPU, MetaData, record_num*key_length_max * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);         //将全部数据由CPU传递给GPU


	i = C[6];
	Init(CPU_Test, i);                            //随机初始化参数
	Error_judge(cudaMemcpy(GPU_parament, CPU_Test, i * sizeof(float), cudaMemcpyHostToDevice), __LINE__);
	cout << "第一层模型开始训练" << endl;

	Stage_One_Train();

	cout << "第一层模型结束" << endl;

	cout << "第二层模型开始训练..." << endl;
	Stage_Two_Train();
	cout << "训练结束，学习索引模型已构建。" << endl;

	cout << "正在批任务查询。" << endl;

	bat_search();

	cout << "完毕" << endl;
	cudaFree(layer1_result);
	cudaFree(layer2_result);
	cudaFree(layer3_result);

	cudaFree(L1);
	cudaFree(L2);
	cudaFree(GPU_parament);
	cudaFree(Model_Parament);
	cudaFree(MetaData);
	cudaFree(result);
	cudaFree(GPU);

	free(CPU);
	free(CPU_MetaData);
	free(CPU_Test);
	cout << "完毕" << endl;
	return 0;

}


__host__ void deleteList(Index_List *op, int num) {              //删除索引结构
	Index_List *mm, *nn;
	int k = 1;
	for (int j = 0;j < num;j++) {
		//cout <<"位置 "<< j <<"  : ";
		mm = op;
		if (mm[j].line == -1) {
			//cout << endl;
			continue;
		}
		nn = mm[j].next;
		k = 1;
		while (nn != NULL) {
			//cout << ++k << " , ";
			mm = nn;
			//cout << nn->next;
			nn = mm->next;
			free(mm);
		}
		//cout << endl;
	}
}

void Init(float *f, int sum) {                          //初始化权重以及常量
	for (int i = 0;i < sum;i++) {
		f[i] = (rand() % 1000)*0.001;                  //随机产生三位精度的小数
	}
}

__host__ void create(int record_num, int bat_num) {                                            // 创建一个数据集，一个批任务
	int p = 0;
	int k = 1;
	int u = 0;
	for (int y = 0;y < key_length_max;y++) {
		p += k;
		k = k * 10;
	}
	ofstream fp(path);
	for (int i = 0; i < record_num; i++)
	{
		u = i + p;
		fp << u <<" "<< endl;
	}
	fp.close();
	ofstream fp_bat(bat_path);
	for (int i = 0; i < bat_num; i++)
	{
		u = i + p;
		fp_bat << u  << " " << endl;
	}
	fp_bat.close();

}

__host__ void deal(float *CPU, char *CPU_MetaData, string s, int i) {                         //读取数据集中的关键字，这里假定以第一列的值作为检索对象，故而直接从第一个字符开始记录
	char str[key_length_max+3];
	strcpy(str, s.c_str());
	int k = 0;
	double A[key_length_max] = { 49,49,49,52.5,52.5,52.5 };
	double B[key_length_max] = {1, 1,1,2.872,2.872,2.872 };
	while (k < key_length_max) {
		//cout <<str[k];
		CPU_MetaData[i] = str[k];
		CPU[i] = (((int)str[k]) - A[k]) / B[k];
		k++;
		i++;
	}
}


__host__ void Stage_One_Train() {                                     //顶层模型处理
	num1 = record_num / oppo;                                         //num1为需要批训练的次数
	num2 = record_num % oppo;
	if (record_num%oppo != 0)
		num1++;
	int i = 0;
	int u = 0;
	int k = 0;
	clock_t start, finish;
	start = clock();
	while (u < 100) {
		i = 0;
		while (i < num1) {                                               //传给metaData第i组数据
			if (i == num1 - 1 && num2 != 0) {
				mini_batch_size = num2;
			}
			else {
				mini_batch_size = oppo;
			}
			ML_Init_Data << <grid, block >> > (MetaData_position, mini_batch_size, i*oppo);                       //训练目标
			Model_Train(mini_batch_size, 0, 0, 0);
			i++;
		}
		u++;
	}

	i = 0;
	//保存第一层模型的参数
	Error_judge(cudaMemcpy(Model_Parament, GPU_parament, C[6] * sizeof(float), cudaMemcpyDeviceToDevice), __LINE__);
	while (i < num1) {
		if (i == num1 - 1 && num2 != 0) {
			mini_batch_size = num2;
		}
		else {
			mini_batch_size = oppo;
		}
		ML_Init_Data << <grid, block >> > (MetaData_position, mini_batch_size, i*oppo);        //训练目标
		Model_forward(mini_batch_size);          //计算最新参数下的结果
		RESULT << <grid, block >> > (layer3_result, result, i*oppo, mini_batch_size);         //划分数据子集
		i++;
	}
	Error_judge(cudaMemcpy(Model_Parament, GPU_parament,  C[6] * sizeof(float), cudaMemcpyDeviceToDevice), __LINE__);    //每个模型的训练初始参数都是第一阶段的训练后参数


}

__host__ void Stage_Two_Train() {
	int i = 0, j = 0;

	//开始分段训练
	Error_judge(cudaMemcpy(CPU, result, record_num * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);    //将划分数据集的依据传递给CPU，在CPU中进行划分
	//每个模型需要训练的数据
	for (i = 0;i < record_num;i++) {
		int hh = CPU[i];
		stageTwo[hh].push_back(i);
	}

	cout << "子数据集划分结束" << endl;
	//开始训练
	float TEST[2] = { 1.0 };
	for (j = 0;j < Stage_Two;j++) {
		int f_num = stageTwo[j].size();        //每个模型元素的个数
		num3[j] = f_num;
		cout << "子数据集 " << j << "     样本个数为  :" << f_num << " :    ";
		for (int u = 0;u < f_num;u++)
			cout << stageTwo[j][u] << "  ,  ";
		cout << endl;
		if (f_num == 0) {           //数据集不存在数据，直接结束
			p[j] = NULL;
			continue;
		}
		cout << "开始训练 ： " << endl;
		int *plk = new int[f_num];

		for (int i = 0;i < f_num;i++) {
			plk[i] = stageTwo[j][i];
		}
		Error_judge(cudaMemcpy(GPU_parament, Model_Parament, C[6] * sizeof(float), cudaMemcpyDeviceToDevice), __LINE__);    //每个模型的训练初始参数都是第一阶段的训练后参数
		num1 = f_num / oppo;         //num1为需要批训练的次数
		num2 = f_num % oppo;
		if (f_num%oppo != 0)
			num1++;
		i = 0;
		int s = 0;
		B[1] = learning_rate / 10;                                                       //每个模型的初始训练率都为learning_rate
		Error_judge(cudaMemcpyToSymbol(parament, B, 2 * sizeof(float)), __LINE__);
		//开始训练
		while (s < 201) {              //最多训练次数,
			if (s % 200 == 0 && s != 0) {
				B[1] = B[1] / 2;
				Error_judge(cudaMemcpyToSymbol(parament, B, 2 * sizeof(float)), __LINE__);
			}
			i = 0;                   //记得每次训练前置0
			while (i < num1) {
				if (i == num1 - 1 && num2 != 0) {
					mini_batch_size = num2;
				}
				else {
					mini_batch_size = oppo;
				}
				Error_judge(cudaMemcpy(MetaData_position, plk + i * oppo, mini_batch_size * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
				Model_Train(mini_batch_size, 0, 0, 0);
				i++;
			}
			if (s % 200 == 0 && s != 0) {         //验证是否训练达标
				TEST[0] = 0.0;
				Error_judge(cudaMemset(GPU, 0, oppo * sizeof(float)), __LINE__);
				i = 0;
				while (i < num1) {
					if (i == num1 - 1 && num2 != 0) {
						mini_batch_size = num2;
					}
					else {
						mini_batch_size = oppo;
					}
					Error_judge(cudaMemcpy(MetaData_position, plk + i * oppo, mini_batch_size * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
					Model_forward(mini_batch_size);
					Model_Judge << <grid, block >> > (MetaData_position, layer3_result, GPU, mini_batch_size);
					i++;
				}
				Sum_Result << <grid, block, ggg_size >> > (GPU, GPU, oppo, 0, 1);
				Sum_Result << <1, 32, lll_size >> > (GPU, GPU, 12, 0, 1);

				Error_judge(cudaMemcpy(TEST, GPU, sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
				//cout << "与训练期望的差值 ：" << TEST[0]<< endl;
				if (TEST[0] == 0)
					break;
			}
			s++;
		}
		if (TEST[0] == 0) {           //训练达标，不需建立索引结构
			Error_judge(cudaMemcpy(CPU_Test, layer3_result, mini_batch_size * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
			cout << "训练成功" << endl;
			cout << "子集的训练结果   :";
			for (i = 0;i < f_num;i++) {
				int f = CPU_Test[i] * record_num;
				cout << f << "  , ";
			}
			cout << endl << endl;
			p[j] = NULL;

		}
		else{
			cout << "训练不达标" << endl << endl;
			i = 0;
			s = 0;
			B[1] = B[1] / 8;
			Error_judge(cudaMemcpyToSymbol(parament, B, 2 * sizeof(float)), __LINE__);
			while (s < 10000) {
				if (s % 200 == 0 && s != 0) {
					B[1] = B[1] / 2;
					Error_judge(cudaMemcpyToSymbol(parament, B, 2 * sizeof(float)), __LINE__);
				}
				while (i < num1) {
					if (i == num1 - 1 && num2 != 0) {
						mini_batch_size = num2;
					}
					else {
						mini_batch_size = oppo;
					}
					Error_judge(cudaMemcpy(MetaData_position, plk + i * oppo, mini_batch_size * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
					Model_Train(mini_batch_size, 1, i*oppo, f_num);
					i++;
				}
				s++;
			}
			i = 0;
			while (i < num1) {                //前向传递
				if (i == num1 - 1 && num2 != 0) {
					mini_batch_size = num2;
				}
				else {
					mini_batch_size = oppo;
				}

				Error_judge(cudaMemcpy(MetaData_position, plk + i * oppo, mini_batch_size * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
				Model_forward(mini_batch_size);
				Deal_index << <grid, block >> > (layer3_result, result, i * oppo, mini_batch_size, f_num);
				i++;
			}
			//cout << "为子数据集 " << j << "  建立索引结构" << endl << endl;
			p[j] = new Index_List[f_num];
			if (p[j] == 0) {
				cout << "结构体空间申请失败" << endl;
			}
			for (i = 0;i < f_num;i++) {                        //初始化生成空间
				p[j][i].key = "";
				p[j][i].line = -1;
				p[j][i].next = NULL;
			}
			Error_judge(cudaMemcpy(CPU, result, f_num * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
			/*cout << "映射位置" << endl;
			for (int uu = 0;uu< f_num;uu++)
				cout << CPU[uu] << endl;
			cout << endl;*/
			int n = 0, m = 0;
			static char pl[key_length_max];
			;
			for (int i = 0;i < f_num;i++) {           //建立索引结构
				n = CPU[i];                     //Hash映射出的数组位置
				//cout <<i<<"对应的位置： "<< n << endl;
				m = stageTwo[j][i];            //键值的信息保存位置
				strncpy(pl, CPU_MetaData + m * key_length_max, key_length_max);        //将记录的键值信息赋给pl
				string l(pl);
				if (p[j][n].line == -1) {
					p[j][n].key = l;                   //赋予键值信息
					p[j][n].line = m;               //赋值该信息所在行数
					p[j][n].next = NULL;
				}
				else {
					Index_List *gg = &p[j][n];
					Index_List *ll = new Index_List;
					if (ll == 0) {
						cout << "内存失败" << __LINE__ << endl;
						exit(0);
					}
					ll->key = l;
					ll->next = NULL;
					ll->line = m;

					while (gg->next != NULL) {
						gg = gg->next;
					}
					gg->next = ll;
				}
			}

		}

		delete[] plk;
		Error_judge(cudaMemcpy(Model_Parament + C[6] + C[6] * j, GPU_parament, C[6] * sizeof(float), cudaMemcpyDeviceToDevice), __LINE__);
	}

}


__host__ void bat_search() {
	cudaFree(Param);
	cudaFree(MetaData);
	cudaFree(A2);
	cudaFree(A1);
	cudaFree(A0);
	cudaFree(BN_o);
	cudaFree(train_test);
	cudaFree(variance);
	cudaFree(result);
	free(CPU);
	free(CPU_MetaData);

	int i = 0;

	CPU = (float *)malloc(bat_num * key_length_max * sizeof(float));
	CPU_MetaData = (char *)malloc(bat_num * key_length_max * sizeof(char));

	if (CPU == NULL || CPU_MetaData == NULL) {
		cout << "ORZ:申请内存失败" << endl;
		exit(0);
	}

	ifstream in_bat(bat_path);
	if (!in_bat) cout << "文件打开异常" << endl;
	string s;
	while (getline(in_bat, s))
	{

		deal(CPU, CPU_MetaData, s, i);

		i = i + key_length_max;
	}
	in_bat.close();
	cout << "文件处理结束！" << endl;
	
	/*****        在GPU中进行批任务的查询   ****/
	//重新生成检索内容的内存空间
	Error_judge(cudaMalloc((void**)&result, bat_num * sizeof(float)), __LINE__);
	Error_judge(cudaMalloc((void**)&MetaData, bat_num*key_length_max * sizeof(float)), __LINE__);

	Error_judge(cudaMemcpy(MetaData, CPU, bat_num*key_length_max * sizeof(float), cudaMemcpyHostToDevice), __LINE__);      //将查询数据传到GPU
	Error_judge(cudaMemcpy(GPU_parament, Model_Parament, C[6] * sizeof(float), cudaMemcpyDeviceToDevice), __LINE__);       //设置模型初始参数
	num1 = bat_num / oppo;
	num2 = bat_num % oppo;
	if (bat_num%oppo != 0)
		num1++;
	i = 0;
	B[1] = learning_rate / 10;
	Error_judge(cudaMemcpyToSymbol(parament, B, 2 * sizeof(float)), __LINE__);
	while (i < num1) {
		if (i == num1 - 1 && num2 != 0) {
			mini_batch_size = num2;
		}
		else {
			mini_batch_size = oppo;
		}

		ML_Init_Data << <grid, block >> > (MetaData_position, mini_batch_size, i*oppo);       //MetaData_position为int型，保存每个记录数据在MetaData中的起始位置/key_length_max
		Model_forward(mini_batch_size);
		RESULT << <grid, block >> > (layer3_result, result, i*oppo, mini_batch_size);
		i++;
	}
	Error_judge(cudaMemcpy(CPU, result, bat_num * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);

	for (i = 0;i < Stage_Two;i++) {           //清空训练中的子集划分，重新将批任务的数据划分
		stageTwo[i].clear();
	}
	for (i = 0;i < bat_num;i++) {
		int hh = CPU[i];                    //hh=CPU[i]保存着该记录应划分于哪个模型
		stageTwo[hh].push_back(i);       //stageTwo[hh]保存着模型hh中的数据对象的位置信息（位于总体查询数据的第几行）
	}
	int j = 0, m = 0, f_num = 0;
	for (j = 0;j < Stage_Two;j++) {              //查询任务按划分的子集分别进行查询
		Error_judge(cudaMemcpy(GPU_parament, Model_Parament + C[6] + C[6] * j, C[6] * sizeof(float), cudaMemcpyDeviceToDevice), __LINE__);      //模型参数传递
		f_num = stageTwo[j].size();        //每个模型元素的个数

		int *plk = new int[f_num];
		for (int i = 0;i < f_num;i++) {
			plk[i] = stageTwo[j][i];
		}

		num1 = f_num / oppo;
		num2 = f_num % oppo;
		if (f_num%oppo != 0)
			num1++;
		i = 0;
		while (i < num1) {
			if (i == num1 - 1 && num2 != 0) {
				mini_batch_size = num2;
			}
			else {
				mini_batch_size = oppo;
			}
			Error_judge(cudaMemcpy(MetaData_position, plk + i * oppo, mini_batch_size * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
			Model_forward(mini_batch_size);
			if (p[j] != NULL)
				Deal_index << <grid, block >> > (layer3_result, result, i*oppo, mini_batch_size, num3[j]);        //将映射函数的值保存在result中
			else Deal_index << <grid, block >> > (layer3_result, result, i*oppo, mini_batch_size, record_num);

			i++;
		}
		delete[] plk;
		Error_judge(cudaMemcpy(CPU, result, f_num * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);     //将映射函数的值传递到CPU中
		static char pl[key_length_max];
		if (p[j] != NULL) {                          //当该划分子集建立了索引结构
			int n = 0;
			for (i = 0;i < f_num;i++) {
				n = CPU[i];                    //记录在索引hh中应该存在的位置为(int) CPU[i]
				m = stageTwo[j][i];            //键值的信息保存位置
				strncpy(pl, CPU_MetaData + m * key_length_max, key_length_max);        //将该记录的键值信息赋给pl
				string l(pl);
				Index_List *mm = &p[j][n];
				while (1) {
					if (mm->key != l) {
						//cout << "匹配键值为 ："<<mm->key << endl;
						if (mm->next == NULL) {
							cout << "键值 " << l << " 无检索结果" << endl;
							break;
						}
						else {
							mm = mm->next;
						}
					}
					else {
						cout << "键值 " << l << "  检索结果为：  " << mm->line << endl;
						break;
					}
				}
			}
		}
		else {
			int n = 0;
			for (i = 0;i < f_num;i++) {
				n = CPU[i];                    //记录在索引hh中应该存在的位置为(int) CPU[i]
				m = stageTwo[j][i];            //键值的信息保存位置
				strncpy(pl, CPU_MetaData + m * key_length_max, key_length_max);        //将该记录的键值信息赋给pl
				string l(pl);
				cout << "键值 " << l << "  检索结果为：  " << n << endl;    //这里不对文件进行读写，故而没有判断该记录是否与计算位置的记录匹配，所以检索结果可能错误
			}

		}
	}

	cout << "批任务检索完毕@！" << endl;

	cout << "删除索引中..." << endl;
	for (int i = 0;i < Stage_Two;i++) {
		if (p[i] != NULL) {
			//cout << "阶段 "<<i << endl;
			deleteList(p[i], num3[i]);
			delete[] p[i];
		}
	}
	for (i = 0;i < Stage_Two;i++) {           //清空子集划分
		stageTwo[i].clear();
	}
}



__global__ void Model_Judge(int *MetaData_position, float *layer3_result, float *result, int length) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;              //线程编号
	int m = 0;
	if (tid < length) {
		m = layer3_result[tid] * constant_parament[3];
		if (m != MetaData_position[tid]) {
			result[tid] += 1;           //是否匹配
		}
	}
}




__host__ void Model_Train(int mini_batch_size, int flag, int i, int num) {
	//cout <<"数据集的大小: "<< mini_batch_size << endl;
	Error_judge(cudaMemset(L1, 0, layer_neru[0] * sizeof(int)), __LINE__);
	Error_judge(cudaMemset(L2, 0, layer_neru[1] * sizeof(int)), __LINE__);
	Model_forward(mini_batch_size);
	Model_reverse(mini_batch_size, flag, i, num);
}

__host__ void Model_forward(int mini_batch_size) {

	BN << <grid, block >> > (MetaData, MetaData_Train, MetaData_position, GPU_parament, mini_batch_size);                      //BN化
	ML_One << <grid, block >> > (MetaData_Train, GPU_parament, layer1_result, mini_batch_size, L1);                         //第一层神经元
	ML_Two << <grid, block >> > (layer1_result, GPU_parament, layer2_result, mini_batch_size, L2);                    //第二层神经元
	ML_Three << <grid, block >> > (layer2_result, GPU_parament, layer3_result, mini_batch_size);                  //第三层神经元

}

__host__ void Model_reverse(int mini_batch_size, int flag, int i, int num) {
	if (flag == 0)
		ML_Variance << <grid, block >> > (layer3_result, MetaData_position, mini_batch_size);   //计算方差
	else ML_Index_Variance << <grid, block >> > (layer3_result, i, mini_batch_size, num);

	/*float A[2048];
	Error_judge(cudaMemcpy(A,layer3_result, mini_batch_size * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
	cout << "目标值为：" << endl;
	for (int i = 0;i < mini_batch_size;i++) {
		int GG = A[i] * record_num;
		cout << "编号为：  " << i << "  , " << A[i] << "   ,  " << GG<<" , "<<(int)A[i] * record_num << endl;

	}

	cout << "结束" << endl;
	//Error_judge(cudaMemcpy(A, layer3_result, mini_batch_size * sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
	int y;
	cin >> y;
	*/
	Sum_Result << <grid, block, ggg_size >> > (layer3_result, GPU, mini_batch_size, 0, 1);  //varience的平均值，保存在variance[0]中
	Sum_Result << <1, 32, lll_size >> > (GPU, variance, 12, 0, 1);

	Error_judge(cudaMemcpy(CPU_Test, variance, sizeof(float), cudaMemcpyDeviceToHost), __LINE__);    //将划分数据集的依据传递给CPU，在CPU中进行划分

	//cout << "偏差为： " << CPU_Test[0] << endl;


	for (int i = 0;i < layer_neru[1];i++) {                       //第二层神经元输入的平均值
		Sum_Result << <grid, block, ggg_size >> > (layer2_result, GPU, mini_batch_size, i, layer_neru[1]); //layer2_result的平均值，保存在A2中
		Sum_Result << <1, 32, lll_size >> > (GPU, A2 + i, 12, 0, 1);;      //总共12个线程块     
	}

	for (int i = 0;i < layer_neru[0];i++) {                       //第一层神经元输入的平均值
		Sum_Result << <grid, block, ggg_size >> > (layer1_result, GPU, mini_batch_size, i, layer_neru[0]); //layer1_result的平均值，保存在A1中，其中L1保存ReLU的导数
		Sum_Result << <1, 32, lll_size >> > (GPU, A1 + i, 12, 0, 1);
	}
	for (int i = 0;i < key_length_max;i++) {                       //BN输入的平均值
		Sum_Result << <grid, block, ggg_size >> > (MetaData_Train, GPU, mini_batch_size, i, key_length_max);   //BN化后训练数据的平均值，保存在A0中
		Sum_Result << <1, 32, lll_size >> > (GPU, A0 + i, 12, 0, 1);;
	}
	for (int i = 0;i < key_length_max;i++) {                       //输入层的平均值
		Sum_Result_BN << <grid, block, ggg_size >> > (MetaData, MetaData_position, GPU, mini_batch_size, i);   //BN前的平均值，保存在BN_L中
		Sum_Result << <1, 32, lll_size >> > (GPU, BN_o + i, 12, 0, 1);
	}
	ML_Param_Update_Before << <1, 1 >> > (GPU_parament, Param, mini_batch_size, L1, L2, variance);     //更新参数前
	ML_Param_Update << <1, 1 >> > (Param, GPU_parament, mini_batch_size, A2, A1, A0, BN_o, variance);        // 参数更新
}




__global__ void ML_Index_Variance(float *layer3_result, int offset, int length, int num) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	float k = 0.0 + offset + tid;
	if (tid < length) {
		layer3_result[tid] = layer3_result[tid] - k / num;
	}

}


__global__ void Deal_index(float *layer3_result, float *result, int offset, int length, int f_num) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;              //线程编号
	int kk = 0, e = layer3_result[tid] / 1;
	float m = layer3_result[tid] - e;  //通用mod(1)函数规范终值范围
	kk = m > 0 ? m * f_num : -m * f_num;
	if (tid < length) {
		result[offset + tid] = kk;          //result中存放着每个记录映射的位置
	}

}



__global__ void RESULT(float *layer3_result, float *result, int offset, int length) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;              //线程编号
	int i = 0, e = 0;
	float f = 0.0, h = 0.0, m = 0.0;
	if (tid < length) {
		result[offset + tid] = 0.0;
		e = layer3_result[tid] / 1;
		m = (layer3_result[tid] - e)*second_model_size[0];  //layer3_result除1取余
		result[offset + tid] = m > 0 ? m : -m;

	}
}

__global__ void ML_Param_Update_Before(float *GPU_parament, float *Param, int mini_batch_size, int *L1, int *L2, float *variance) {        //链式求导法则推导
	variance[0] = variance[0] / mini_batch_size;
	int i = 0, n = 0, f = 0, gg = 0, kk = 0;
	int m = parament_offset[2];     //第二层神经元的导数
	for (;i < constant_parament[2];i++) {
		Param[i] = 0.0;
		Param[i] = GPU_parament[m + i] * (L2[i] * 0.001 + mini_batch_size - L2[i]) / mini_batch_size;        //L2表示ReLU的导数,这里 需要使用平均值：L2[i]/mini_batch_size
	}
	m = parament_offset[1];        //第一层神经元的导数
	for (;f < constant_parament[1];f++) {
		Param[i + f] = 0.0;
		for (;n < constant_parament[2];n++) {
			gg = n * constant_parament[1];
			Param[i + f] += Param[n] * GPU_parament[m + gg + f] * (L1[f] * 0.001 + mini_batch_size - L1[f]) / mini_batch_size;
		}
	}
	i = i + f;
	m = parament_offset[0];        //BN层神经元的导数
	n = 0;
	f = 0;
	gg = 0;
	kk = constant_parament[2];
	for (;f < constant_parament[0];f++) {
		Param[i + f] = 0.0;
		for (;n < constant_parament[1];n++) {
			gg = n * constant_parament[0];
			Param[i + f] += Param[kk + n] * GPU_parament[m + gg + f];
		}
	}
}

__global__ void ML_Param_Update(float *Param, float *GPU_parament, int mini_batch_size, float *A2, float *A1, float *A0, float *BN_o, float *variance) {
	int i = 0, n = 0, gg = 0;
	int m = parament_offset[2];     //输出层参数
	int k = parament_offset[5];
	for (;i < constant_parament[2];i++) {
		GPU_parament[m + i] = GPU_parament[m + i] - parament[1] * A2[i] * variance[0] / mini_batch_size;
		GPU_parament[k + i] = GPU_parament[k + i] - parament[1] * variance[0];
	}

	m = parament_offset[1];        //第二层神经元
	k = parament_offset[4];
	i = 0;n = 0;
	for (;i < constant_parament[2];i++) {
		gg = i * constant_parament[1];
		for (;n < constant_parament[1];n++) {
			GPU_parament[m + gg + n] = GPU_parament[m + gg + n] - parament[1] * A1[n] * Param[i] * variance[0] / mini_batch_size;
		}
		GPU_parament[k + i] = GPU_parament[k + i] - parament[1] * Param[i] * variance[0];
	}
	m = parament_offset[0];        //第一层神经元
	k = parament_offset[3];
	i = 0;n = 0;
	int f = constant_parament[2];
	for (;i < constant_parament[1];i++) {
		gg = n * constant_parament[0];
		for (;n < constant_parament[0];n++) {
			GPU_parament[m + gg + n] = GPU_parament[m + gg + n] - parament[1] * A0[n] * Param[f + i] * variance[0] / mini_batch_size;
		}
		GPU_parament[k + i] = GPU_parament[k + i] - parament[1] * Param[f + i] * variance[0];
	}
	m = constant_parament[2] + constant_parament[1];        //第零层参数（BN参数）
	k = constant_parament[0];
	k = 2;
	i = 0;
	for (;i < constant_parament[0];i++) {
		GPU_parament[i] = GPU_parament[i] - parament[1] * BN_o[i] * Param[m + i] * variance[0] / mini_batch_size;
		GPU_parament[i + k] = GPU_parament[i + k] - parament[1] * Param[m + i] * variance[0];
	}

}



__global__ void Sum_Result_BN(float *input, int *position, float *sum_result, int length, int i) {
	int threadId = threadIdx.x + threadIdx.y*blockDim.x;
	int blockId = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = threadId + (blockDim.x*blockDim.y)*blockId;
	int f = position[tid] * constant_parament[0];
	extern __shared__ float sdata[];
	float x = 0.0;
	if (tid < length)
	{
		x = input[f + i];
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

__global__ void	Sum_Result(float *input, float *sum_result, int length, int i, int offset) {
	//const int a = blockDim.x*blockDim.y;
	int threadId = threadIdx.x + threadIdx.y*blockDim.x;
	int blockId = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = threadId + (blockDim.x*blockDim.y)*blockId;
	extern __shared__ float sdata[];
	int f = offset * tid;
	float x = 0;
	if (tid < length)
	{
		x = input[f + i];
	}
	sdata[threadId] = x;
	__syncthreads();          //等待所有线程把自己负责的元素载入到共享内存

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

__global__ void ML_Init_Data(int *MetaData_position, int mini_batch_size, int offset) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	MetaData_position[tid] = 0;
	if (tid < mini_batch_size) {
		MetaData_position[tid] = offset + tid;
	}

}


__global__ void BN(float *MetaData, float *MetaData_Train, int *MetaData_position, float *GPU_parament, int length) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int k = MetaData_position[tid] * constant_parament[0];       //得到记录在MetaData的起始位置
	int i = 0, j = tid * constant_parament[0], s = 0;
	for (;i < constant_parament[0];i++) {
		s = i + constant_parament[0];
		MetaData_Train[j + i] = 0.0;
		if (tid < length)
			MetaData_Train[j + i] =  MetaData[k + i]* GPU_parament[i]+GPU_parament[s+i];          //得到类BN化后的新数据
	}
}

__global__ void ML_One(float *MetaData_Train, float *GPU_parament, float *layer1_result, int length, int *L1) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;              //线程编号
	int offset1 = tid * constant_parament[0];
	int offset2 = tid * constant_parament[1];
	int k = 0, m = 0, f = 0, t = parament_offset[0], b = parament_offset[3];
	for (;k < constant_parament[1];k++) {                    //每个样本有constant_parament[1]个结果
		f = k * constant_parament[0];                       //切换到下一个神经元的参数的起始位置
		layer1_result[offset2 + k] = 0.0;                 //初始化为0
		if (tid < length) {
			m = 0;
			for (;m < constant_parament[0];m++) {
				layer1_result[offset2 + k] += MetaData_Train[offset1 + m] * GPU_parament[t + f + m];
			}
			layer1_result[offset2 + k] += GPU_parament[b + k];                 //一个神经元的计算值
			if (layer1_result[offset2 + k] < 0)                        //ReLU激活函数
			{
				layer1_result[offset2 + k] = 0.001*layer1_result[offset2 + k];
				atomicAdd(&L1[m], 1);
			}
		}
	}
}

__global__ void	ML_Two(float *layer1_result, float *GPU_parament, float *layer2_result, int length, int *L2) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int offset1 = constant_parament[1] * tid;
	int offset2 = constant_parament[2] * tid;
	int k = 0, m = 0, f = 0, t = parament_offset[1], b = parament_offset[4];
	for (;k < constant_parament[2];k++) {              // 每个样本有constant_parament[2]个结果
		f = k * constant_parament[1];
		layer2_result[offset2 + k] = 0.0;               //初始化为0.0
		if (tid < length) {
			m = 0;
			for (;m < constant_parament[1];m++) {
				layer2_result[offset2 + k] += layer1_result[offset1 + m] * GPU_parament[t + f + m];
			}
			layer2_result[offset2 + k] += GPU_parament[b + k];
			if (layer2_result[offset2 + k] < 0) {
				atomicAdd(&L2[m], 1);
				layer2_result[offset2 + k] = 0.001*layer1_result[offset2 + k];
			}
		}
	}

}

__global__ void	ML_Three(float *layer2_result, float *GPU_parament, float *layer3_result, int length) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int offset1 = constant_parament[2] * tid;
	int offset2 = 1 * tid;
	int m = 0, t = parament_offset[2], b = parament_offset[5];
	layer3_result[offset2] = 0.0;              //初始化为0.0

	if (tid < length) {
		for (;m < constant_parament[2];m++) {
			layer3_result[offset2] += layer2_result[offset1 + m] * GPU_parament[t + m];
		}
		layer3_result[offset2] += GPU_parament[b];
	}
}

__global__ void ML_Variance(float *layer3_result, int *MetaData_position, int length) {
	int tid = (blockIdx.x + blockIdx.y*gridDim.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	float k = 1.0*MetaData_position[tid];
	if (tid < length) {
		layer3_result[tid] = layer3_result[tid]- k / constant_parament[3];
	}
}

