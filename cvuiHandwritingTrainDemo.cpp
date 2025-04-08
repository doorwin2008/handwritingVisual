/*
* 版权所有 东海仙人岛 2025年3月 
* B站：https://space.bilibili.com/627167269
* 功能需求：
* 手写数组输入功能，可视化展示机器学习神经网络的结果，模拟神经元链接，动态展示识别过程的变化
* 设计方案：
* 界面显示，采用cvui; 权重文件的读取，采用opencv 4.10.0
* 网络结构，输入层 28*28，隐藏层 64 ，输出层 10
* 激活函数 y=(1. - x) / (1. + x)
* 使用说明：
* 权重文件，采用MNIST训练的数据集
*左键书写，右键擦除，esc按键退出.左侧的复选框可以显示或隐藏神经元连线。
*2025-3-8 更新记录：位于之前用32为float数，导致负无穷小和无穷大的数判断出现问题，修改为64位数参与计算。
*/
#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cstring>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CThread.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

// One (and only one) of your C++ files must define CVUI_IMPLEMENTATION
// before the inclusion of cvui.h to ensure its implementaiton is compiled.
#define CV_TERMCRIT_ITER    1
#define CV_TERMCRIT_NUMBER  CV_TERMCRIT_ITER
#define CV_TERMCRIT_EPS     2
#define CVUI_IMPLEMENTATION

/** Train options */
enum TrainFlags {
	/** Update the network weights, rather than compute them from scratch. In the latter case
	the weights are initialized using the Nguyen-Widrow algorithm. */
	UPDATE_WEIGHTS = 1,
	/** Do not normalize the input vectors. If this flag is not set, the training algorithm
	normalizes each input feature independently, shifting its mean value to 0 and making the
	standard deviation equal to 1. If the network is assumed to be updated frequently, the new
	training data could be much different from original one. In this case, you should take care
	of proper normalization. */
	NO_INPUT_SCALE = 2,
	/** Do not normalize the output vectors. If the flag is not set, the training algorithm
	normalizes each output feature independently, by transforming it to the certain range
	depending on the used activation function. */
	NO_OUTPUT_SCALE = 4
};

#include "cvui.h"
#include "cv_puttextzh.h"

#define WINDOW_NAME "机器学习手写数字输入可视化"
struct draw_s {
	int x;
	int y;
	float v;
};
struct draw_sHistory {
	int x;
	int y;
	float v;
	float vOld;
};
//调色板
uchar doubleColorValue_R = 31;
uchar doubleColorValue_G = 131;
uchar doubleColorValue_B = 231;
uchar doubleColorValue_A = 128;
unsigned int  bdColor;
unsigned int  filColor;
//书写方格子
draw_s g_inputLayerDraw[28*28];
cv::Rect g_inRect = { 400,480,7,7 };
//展开格子显示
draw_s g_inputLayerDraw784[28 * 28];
//隐藏层坐标和权重信息，64个计算结果
draw_s g_hidLayerDraw[64];
draw_sHistory g_hidLayerWeights[65][10];
//输出层坐标和权重信息，10个计算结果
draw_s g_outputLayerDraw[10];
int g_mouse_x;
int g_mouse_y;
cv::String g_ResultString = "";
int g_gradw = 30;
int g_gradh = 50;

uint8_t g_mouseKeyleftRight = 0;
//定义输出buffer
cv::AutoBuffer<double> _buf(1568 + 10);
//结果输出矩阵
Mat outputs;

// for train 
cv::Ptr<cv::ml::ANN_MLP> ann;
//小端存储转换
int reverseInt(int i);
//读取image数据集信息
Mat read_mnist_image(const string fileName);
//读取label数据集信息
Mat read_mnist_label(const string fileName);
//将标签数据改为one-hot型
Mat one_hot(Mat label, int classes_num);
vector<int> layer_sizes= { 784,64,10 };;
double min_val, max_val, min_val1, max_val1;
vector<Mat> weights ;
RNG rng;
int paramstermCrittype = 3;
bool trained;
int max_lsize=784;
int g_train_cycle = 0;
long g_ErrorDeviation = 0;
float g_each_number_error[10] ;
float g_each_number_error2[10];
int g_train_count = 1;
string train_images_path = "train-images.idx3-ubyte";
string train_labels_path = "train-labels.idx1-ubyte";

int layer_count() { return (int)layer_sizes.size(); }
int g_train_sleep_time = 1000;

int g_thickness = 2;
float tempW;
float filter_step = 0.0001;
//-----------------------------------thread
CMutex g_metux;
int startTrain();


class TrainThread : public CThread
{
public:
	TrainThread(const std::string& strName)
		: m_strThreadName(strName)
	{
		cout << "train thread start." << endl;
	}

	~TrainThread()
	{
	}

public:
	virtual void Run()
	{
		startTrain(); //另外一个线程中启动训练开始
	}
private:
	std::string m_strThreadName;
};

//-----------------------------------thread
void init_weights()
{
	int i, j, k, l_count = layer_count();

	for (i = 1; i < l_count; i++)
	{
		int n1 = layer_sizes[i - 1];
		int n2 = layer_sizes[i];
		double val = 0, G = n2 > 2 ? 0.7 * pow((double)n1, 1. / (n2 - 1)) : 1.;
		double* w = weights[i].ptr<double>();

		// initialize weights using Nguyen-Widrow algorithm
		for (j = 0; j < n2; j++)
		{
			double s = 0;
			for (k = 0; k <= n1; k++)
			{
				val = rng.uniform(0., 1.) * 2 - 1.;
				w[k * n2 + j] = val;
				s += fabs(val);
			}

			if (i < l_count - 1)
			{
				s = 1. / (s - fabs(val));
				for (k = 0; k <= n1; k++)
					w[k * n2 + j] *= s;
				w[n1 * n2 + j] *= G * (-1 + j * 2. / n2);
			}
		}
	}
}

void calc_input_scale(const Mat& inputs, int flags)
{
	bool reset_weights = (flags & UPDATE_WEIGHTS) == 0;
	bool no_scale = (flags & NO_INPUT_SCALE) != 0;
	double* scale = weights[0].ptr<double>();
	int count = inputs.rows;

	if (reset_weights)
	{
		int i, j, vcount = layer_sizes[0];
		int type = inputs.type();
		double a = no_scale ? 1. : 0.;

		for (j = 0; j < vcount; j++)
			scale[2 * j] = a, scale[j * 2 + 1] = 0.;

		if (no_scale)
			return;

		for (i = 0; i < count; i++)
		{
			const uchar* p = inputs.ptr(i);
			const float* f = (const float*)p;
			const double* d = (const double*)p;
			for (j = 0; j < vcount; j++)
			{
				double t = type == CV_32F ? (double)f[j] : d[j];
				scale[j * 2] += t;
				scale[j * 2 + 1] += t * t;
			}
		}

		for (j = 0; j < vcount; j++)
		{
			double s = scale[j * 2], s2 = scale[j * 2 + 1];
			double m = s / count, sigma2 = s2 / count - m * m;
			scale[j * 2] = sigma2 < DBL_EPSILON ? 1 : 1. / sqrt(sigma2);
			scale[j * 2 + 1] = -m * scale[j * 2];
		}
	}
}

void calc_output_scale(const Mat& outputs, int flags)
{
	int i, j, vcount = layer_sizes.back();
	int type = outputs.type();
	double m = min_val, M = max_val, m1 = min_val1, M1 = max_val1;
	bool reset_weights = (flags & UPDATE_WEIGHTS) == 0;
	bool no_scale = (flags & NO_OUTPUT_SCALE) != 0;
	int l_count = layer_count();
	double* scale = weights[l_count].ptr<double>();
	double* inv_scale = weights[l_count + 1].ptr<double>();
	int count = outputs.rows;

	if (reset_weights)
	{
		double a0 = no_scale ? 1 : DBL_MAX, b0 = no_scale ? 0 : -DBL_MAX;

		for (j = 0; j < vcount; j++)
		{
			scale[2 * j] = inv_scale[2 * j] = a0;
			scale[j * 2 + 1] = inv_scale[2 * j + 1] = b0;
		}

		if (no_scale)
			return;
	}

	for (i = 0; i < count; i++)
	{
		const uchar* p = outputs.ptr(i);
		const float* f = (const float*)p;
		const double* d = (const double*)p;

		for (j = 0; j < vcount; j++)
		{
			double t = type == CV_32F ? (double)f[j] : d[j];

			if (reset_weights)
			{
				double mj = scale[j * 2], Mj = scale[j * 2 + 1];
				if (mj > t) mj = t;
				if (Mj < t) Mj = t;

				scale[j * 2] = mj;
				scale[j * 2 + 1] = Mj;
			}
			else if (!no_scale)
			{
				t = t * inv_scale[j * 2] + inv_scale[2 * j + 1];
				if (t < m1 || t > M1)
					CV_Error(cv::Error::StsOutOfRange,
						"Some of new output training vector components run exceed the original range too much");
			}
		}
	}

	if (reset_weights)
		for (j = 0; j < vcount; j++)
		{
			// map mj..Mj to m..M
			double mj = scale[j * 2], Mj = scale[j * 2 + 1];
			double a, b;
			double delta = Mj - mj;
			if (delta < DBL_EPSILON)
				a = 1, b = (M + m - Mj - mj) * 0.5;
			else
				a = (M - m) / delta, b = m - mj * a;
			inv_scale[j * 2] = a; inv_scale[j * 2 + 1] = b;
			a = 1. / a; b = -b * a;
			scale[j * 2] = a; scale[j * 2 + 1] = b;
		}
}
void prepare_to_train(const Mat& inputs, const Mat& outputs,
	Mat& sample_weights, int flags)
{
	if (layer_sizes.empty())
		CV_Error(cv::Error::StsError,
			"The network has not been created. Use method create or the appropriate constructor");

	if ((inputs.type() != CV_32F && inputs.type() != CV_64F) ||
		inputs.cols != layer_sizes[0])
		CV_Error(cv::Error::StsBadArg,
			"input training data should be a floating-point matrix with "
			"the number of rows equal to the number of training samples and "
			"the number of columns equal to the size of 0-th (input) layer");

	if ((outputs.type() != CV_32F && outputs.type() != CV_64F) ||
		outputs.cols != layer_sizes.back())
		CV_Error(cv::Error::StsBadArg,
			"output training data should be a floating-point matrix with "
			"the number of rows equal to the number of training samples and "
			"the number of columns equal to the size of last (output) layer");

	if (inputs.rows != outputs.rows)
		CV_Error(cv::Error::StsUnmatchedSizes, "The numbers of input and output samples do not match");

	Mat temp;
	double s = sum(sample_weights)[0];
	sample_weights.convertTo(temp, CV_64F, 1. / s);
	sample_weights = temp;

	calc_input_scale(inputs, flags);
	calc_output_scale(outputs, flags);
}
//参数：
//_xf：一个 Mat 类型的引用，通常表示输入特征矩阵，在函数执行过程中会被修改。
//_df：一个 Mat 类型的引用，用于存储激活函数导数的偏移量，在函数执行过程中会被修改。
//w：一个 const Mat 类型的引用，通常表示权重矩阵，函数执行过程中不会修改它。
void calc_activ_func_deriv(Mat& _xf, Mat& _df, const Mat& w)
{
	//获取偏置值：
	const double* bias = w.ptr<double>(w.rows - 1);
	//获取输入特征矩阵 _xf 的行数 n 和列数 cols，并初始化两个缩放因子 scale 和 scale2。
	int i, j, n = _xf.rows, cols = _xf.cols;
	{
		double scale = 1.0;
		double scale2 = 1.0;
		//第一次循环处理输入特征和导数矩阵：
		for (i = 0; i < n; i++)
		{
			double* xf = _xf.ptr<double>(i);
			double* df = _df.ptr<double>(i);

			for (j = 0; j < cols; j++)
			{
				//对输入特征矩阵 _xf 的每一个元素加上对应的偏置值，然后乘以缩放因子 scale
				xf[j] = (xf[j] + bias[j]) * scale;
				df[j] = -fabs(xf[j]);//求浮点数的 绝对值。
			}
		}
		// _df 矩阵的每一个元素计算 e 的幂次方
		exp(_df, _df);

		// ((1+exp(-ax))^-1)'=a*((1+exp(-ax))^-2)*exp(-ax);
		// ((1-exp(-ax))/(1+exp(-ax)))'=(a*exp(-ax)*(1+exp(-ax)) + a*exp(-ax)*(1-exp(-ax)))/(1+exp(-ax))^2=
		// 2*a*exp(-ax)/(1+exp(-ax))^2
		scale *= 2 * 1.0;//更新缩放因子

		//第二次循环处理输入特征和导数矩阵
		//计算激活函数的导数并更新 _df 矩阵，同时更新 _xf 矩阵的值。
		//根据 xf[j] 的正负确定符号 s0，然后计算中间变量 t0 和 t1，
		//最后更新 _df 和 _xf 矩阵的对应元素
		//激活函数： y = (1 - e ^ x )/（1 + e ^ x)的导数函数为： y' = （ -2e ^ x ）/(1 + e ^ x) ^ 2
		for (i = 0; i < n; i++)
		{
			double* xf = _xf.ptr<double>(i);
			double* df = _df.ptr<double>(i);

			for (j = 0; j < cols; j++)
			{
				int s0 = xf[j] > 0 ? 1 : -1;
				double t0 = 1. / (1. + df[j]);
				double t1 = scale * df[j] * t0 * t0;
				t0 *= scale2 * (1. - df[j]) * s0;
				df[j] = t1;
				xf[j] = t0;
			}
		}
	}
}
/*该函数实现了神经网络的反向传播训练算法，通过不断调整权重来最小化误差。
具体步骤包括前向传播计算输出，计算误差，反向传播更新权重，直到满足终止条件。
* 参数说明
const Mat& inputs：输入数据矩阵，每一行代表一个输入样本。
const Mat& outputs：期望输出数据矩阵，每一行对应一个输入样本的期望输出。
const Mat& _sw：样本权重矩阵，用于对不同样本的误差进行加权。
TermCriteria termCrit：训练终止条件，包含最大迭代次数和误差阈值。
*/
int train_backprop(const Mat& inputs, const Mat& outputs, const Mat& _sw, TermCriteria termCrit)
{
	int i, j, k;
	//prev_E 8.9884656743115785e+307 和 E：分别存储上一次迭代和当前迭代的误差。
	double prev_E = DBL_MAX * 0.5, E = 0; // DBL_MAX 1.7976931348623158e+308
	//itype 和 otype：分别存储输入和输出矩阵的数据类型。
	int itype = inputs.type(), otype = outputs.type();
	//count：输入样本的数量。
	int count = inputs.rows;

	//max_iter：最大迭代次数。
	int iter = -1, max_iter = termCrit.maxCount * count; 
	//epsilon =6：误差阈值。CV_TERMCRIT_EPS=2
	double epsilon = (termCrit.type & CV_TERMCRIT_EPS) ? termCrit.epsilon * count : 0;
	//l_count：神经网络的层数。
	int l_count = layer_count();
	//ivcount：输入层的神经元数量。
	int ivcount = layer_sizes[0];
	//ovcount：输出层的神经元数量。
	int ovcount = layer_sizes.back();

	// allocate buffers
	vector<vector<double> > x(l_count);
	vector<vector<double> > df(l_count);
	// 存储每一层权重的更新量
	vector<Mat> dw(l_count);
	// 分配缓冲区
	for (i = 0; i < l_count; i++)
	{
		int n = layer_sizes[i];
		x[i].resize(n + 1);//x：存储每一层的输出。
		df[i].resize(n);//df：存储每一层激活函数的导数。
		dw[i] = Mat::zeros(weights[i].size(), CV_64F);//dw：存储每一层权重的更新量。
	}
	//初始化样本索引
	Mat _idx_m(1, count, CV_32S);
	int* _idx = _idx_m.ptr<int>();//_idx_m：存储样本的索引，用于随机打乱样本顺序。
	for (i = 0; i < count; i++)
		_idx[i] = i;
	//初始化缓冲区
	AutoBuffer<double> _buf(max_lsize * 2);//_buf：用于存储临时数据。
	double* buf[] = { _buf.data(), _buf.data() + max_lsize };//sw：样本权重矩阵的指针。

	const double* sw = _sw.empty() ? 0 : _sw.ptr<double>();

	// run back-propagation loop
	/*
	 y_i = w_i*x_{i-1}
	 x_i = f(y_i)
	 E = 1/2*||u - x_N||^2
	 grad_N = (x_N - u)*f'(y_i)
	 dw_i(t) = momentum*dw_i(t-1) + dw_scale*x_{i-1}*grad_i
	 w_i(t+1) = w_i(t) + dw_i(t)
	 grad_{i-1} = w_i^t*grad_i
	*/
	//反向传播训练循环，总的图片的个数，60000 max_iter
	for (iter = 0; iter < max_iter; iter++)
	{
		//doorwin 复制每张图片的原始数据，用于手写笔显示
		memset(g_inputLayerDraw, 0, sizeof(g_inputLayerDraw));
		cout << "Thread Train==============> " << iter <<"/"<< max_iter << endl;
		g_train_cycle = iter;
		const uchar* ppp = inputs.ptr(iter % count);
		const float* ccc = (const float*)ppp;
		for (int m = 0; m < 784; m++)
		{
			if (ccc[m] > 0)g_inputLayerDraw[m].v = 255;//用来显示数字到面板
		}
		//输出0-9的刷新，通过读出来的标签信息
		const uchar* ddd = outputs.ptr(iter % count);
		const float* eee = (const float*)ddd;
		memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));
		for (int n = 0; n < 10; n++)
		{
			if (eee[n] == 1)g_outputLayerDraw[n].v = 1;
		}
		Sleep(g_train_sleep_time); //子线程挂起，演示页面刷新慢一点
		//doorwin
		//处理单个样本
		int idx = iter % count; //idx：当前处理的样本索引。
		double sweight = sw ? count * sw[idx] : 1.;//当前样本的权重。

		//当处理完一轮所有样本后，检查误差是否小于阈值，如果是则停止训练。同时，随机打乱样本顺序。
		if (idx == 0)
		{
			//printf("%d. E = %g\n", iter/count, E);
			if (fabs(prev_E - E) < epsilon)
				break;//达到训练目标，结束训练，退出
			prev_E = E;
			E = 0;

			// shuffle indices 随机索引
			for (i = 0; i < count; i++)
			{
				j = rng.uniform(0, count);
				k = rng.uniform(0, count);
				std::swap(_idx[j], _idx[k]);
			}
		}
		//前向传播
		idx = _idx[idx];

		const uchar* x0data_p = inputs.ptr(idx);
		const float* x0data_f = (const float*)x0data_p;
		const double* x0data_d = (const double*)x0data_p;

		double* w = weights[0].ptr<double>();
		//每个图片逐个像素计算
		for (j = 0; j < ivcount; j++)
		{
			x[0][j] = (itype == CV_32F ? (double)x0data_f[j] : x0data_d[j]) * w[j * 2] + w[j * 2 + 1];
		}	

		Mat x1(1, ivcount, CV_64F, &x[0][0]);

		//从输入层开始，逐层计算每一层的输出 x 和激活函数的导数 df。
		// forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
		for (i = 1; i < l_count; i++)
		{
			int n = layer_sizes[i];
			Mat x2(1, n, CV_64F, &x[i][0]);
			Mat _w = weights[i].rowRange(0, x1.cols);//读取 0~784  ； 0~64 的值
			gemm(x1, _w, 1, noArray(), 0, x2);//矩阵乘 //前三个参数相乘 4 = 1*2*3
			Mat _df(1, n, CV_64F, &df[i][0]);//df存储每一层权重的更新量
			calc_activ_func_deriv(x2, _df, weights[i]);//激活函数导数,通过导数计算，返回更新了df，导入函数带入x之后的运算结果
			//cout <<"_df : i= "<<to_string(i) << endl;
			if (i == 1)
			{
				for (int d = 0; d < 64; d++)
				{
					g_hidLayerDraw[d].v = _df.at<double>(0, d);
				}
			}
			else if (i == 2)
			{
				for (int d = 0; d < _df.cols; d++)
				{
					//cout << _df.at<double>(0, d)<< "\t"  ;
					g_each_number_error2[d] = _df.at<double>(0, d);
				}
				//cout <<  endl;
			}
			x1 = x2;
		}
		//. 计算误差 
		Mat grad1(1, ovcount, CV_64F, buf[l_count & 1]);
		w = weights[l_count + 1].ptr<double>();

		// calculate error
		const uchar* udata_p = outputs.ptr(idx);
		const float* udata_f = (const float*)udata_p;
		const double* udata_d = (const double*)udata_p;

		double* gdata = grad1.ptr<double>();
		cout << endl;
		for (k = 0; k < ovcount; k++)
		{
			double t = (otype == CV_32F ? (double)udata_f[k] : udata_d[k]) * w[k * 2] + w[k * 2 + 1] - x[l_count - 1][k];
			gdata[k] = t * sweight;
			E += t * t;
		}
		E *= sweight; //计算输出层的误差 grad1，并累加误差 E。
		g_ErrorDeviation = E;
		//cout << endl << " E= " << E << endl;

		// 反向传播 backward pass, update weights
		//从输出层开始，逐层更新权重 weights 和误差 grad1。
		//multiply 函数用于元素级乘法。
		//add 函数用于矩阵加法。
		for (i = l_count - 1; i > 0; i--)
		{
			int n1 = layer_sizes[i - 1], n2 = layer_sizes[i];
			Mat _df(1, n2, CV_64F, &df[i][0]);
			multiply(grad1, _df, grad1);//基于元素的乘法计算 3 = 1*2
			Mat _x(n1 + 1, 1, CV_64F, &x[i - 1][0]);
			x[i - 1][n1] = 1.;
			//前三个参数相乘 4 = 1*2*3
			gemm(_x, grad1,0.01 , dw[i], 0.1, dw[i]);////矩阵乘params.bpDWScale  params.bpMomentScale
			//前两个参数相加 3=1+2
			add(weights[i], dw[i], weights[i]);
			if (i > 1)
			{
				Mat grad2(1, n1, CV_64F, buf[i & 1]);
				Mat _w = weights[i].rowRange(0, n1);
				gemm(grad1, _w, 1, noArray(), 0, grad2, GEMM_2_T);//矩阵乘
				grad1 = grad2;
			}
		}
		//输出weights信息给HMI显示
		int tp = weights[2].type();
		for (int j = 0; j < 65; j++)
			for (int i = 0; i < 10; i++)
			{
				g_hidLayerWeights[j][i].v = (float)weights[2].at<double>(j, i);
			}
		//统计误差
		//cout<<"gdata" << endl;
		for (int i = 0; i < 10; i++)
		{
			g_each_number_error[i] = gdata[i];
			//cout << gdata[i] << "\t";
		}
		//cout  << endl;
	}//end for 所有图片
	
	//返回迭代次数
	iter /= count;
	return iter;
}

bool HWtrain(const Ptr<TrainData>& trainData, int flags)
{
	CV_Assert(!trainData.empty());
	const int MAX_ITER = 1000;
	const double DEFAULT_EPSILON = FLT_EPSILON;

	// initialize training data
	Mat inputs = trainData->getTrainSamples();
	Mat outputs = trainData->getTrainResponses();
	Mat sw = trainData->getTrainSampleWeights();

	prepare_to_train(inputs, outputs, sw, flags);

	// ... and link weights
	if (!(flags & 1))
		init_weights();

	TermCriteria termcrit;
	termcrit.type = TermCriteria::COUNT + TermCriteria::EPS;
	termcrit.maxCount = g_train_count;// std::max((paramstermCrittype & CV_TERMCRIT_ITER ? 1 : MAX_ITER), 1);
	termcrit.epsilon = std::max((paramstermCrittype & CV_TERMCRIT_EPS ? 0.0001 : DEFAULT_EPSILON), DBL_EPSILON);

	int iter = 0;

	iter = train_backprop(inputs, outputs, sw, termcrit);

	trained = iter > 0;
	return trained;
}

int startTrain()
{
	//system("chcp 65001");
	/*
	---------第一部分：训练数据准备-----------
	*/
	//读取训练标签数据 (60000,1) 类型为int32
	Mat train_labels = read_mnist_label(train_labels_path);
	if (train_labels.empty()) {
		cerr << "Failed to read training labels." << endl;
		return -1;
	}
	//ann神经网络的标签数据需要转为one-hot型 编码用于将离散的分类标签转换为二进制向量。
	train_labels = one_hot(train_labels, 10);

	//读取训练图像数据 (60000,784) 类型为float32 数据未归一化
	Mat train_images = read_mnist_image(train_images_path);
	if (train_images.empty()) {
		cerr << "Failed to read training images." << endl;
		return -1;
	}
	//将图像数据归一化
	train_images = train_images / 255.0;

	/*
	---------第二部分：构建ann训练模型并进行训练-----------
	*/
	ann = cv::ml::ANN_MLP::create();
	//定义模型的层次结构 输入层为784 隐藏层为64 输出层为10
	Mat layerSizes = (Mat_<int>(1, 3) << 784, 64, 10);
	//初始化weights数据
	ann->setLayerSizes(layerSizes);
	//设置参数更新为误差反向传播法
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	//设置激活函数为sigmoid 
	//对称sigmoid函数（ANN_MLP::SIGMOID_SYM）：MLP默认函数，  
	//𝑓(𝑥) = 𝛽∗(1−𝑒 ^ −𝛼𝑥) / (1 + 𝑒 ^ −𝛼𝑥)。 β = 1, α = 1  时为标准sigmoid函数
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	//设置跌打条件 最大训练次数为100
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 0.0001));
	Mat ts = ann->getLayerSizes() ;
	for (int i = 0; i < ts.rows +2; i++) 
	{
		weights.push_back( ann->getWeights(i));//这里返回的只是地址，weights的数据是同一份
		//weights[i] = ann->getWeights(i);
	}
	//开始训练
	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_images, cv::ml::ROW_SAMPLE, train_labels);
	cout << "开始进行训练..." << endl;
	if (!HWtrain(train_data,0)) {
		cerr << "Training failed." << endl;
		return -1;
	}
	cout << "训练完成" << endl;

	//保存模型
	ann->save("mnist_ann.xml");

	cout << "按键退出" << endl;
	getchar();
	return 0;
}

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat read_mnist_image(const string fileName) {
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	Mat DataMat;

	ifstream file(fileName, ios::binary);
	if (file.is_open())
	{
		cout << "成功打开图像集 ..." << endl;

		file.read((char*)&magic_number, sizeof(magic_number));//幻数（文件格式）
		if (!file.good()) {
			cerr << "Error reading magic number from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&number_of_images, sizeof(number_of_images));//图像总数
		if (!file.good()) {
			cerr << "Error reading number of images from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&n_rows, sizeof(n_rows));//每个图像的行数
		if (!file.good()) {
			cerr << "Error reading number of rows from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&n_cols, sizeof(n_cols));//每个图像的列数
		if (!file.good()) {
			cerr << "Error reading number of columns from " << fileName << endl;
			return Mat();
		}

		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
		cout << "幻数（文件格式）:" << magic_number
			<< " 图像总数:" << number_of_images
			<< " 每个图像的行数:" << n_rows
			<< " 每个图像的列数:" << n_cols << endl;

		cout << "开始读取Image数据......" << endl;

		DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
		for (int i = 0; i < number_of_images; i++) {
			for (int j = 0; j < n_rows * n_cols; j++) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				if (!file.good()) {
					cerr << "Error reading pixel data from " << fileName << " at row " << i << ", col " << j << endl;
					return Mat();
				}
				//可以在下面这一步将每个像素值归一化
				float pixel_value = float(temp);
				//按照行将像素值一个个写入Mat中
				DataMat.at<float>(i, j) = pixel_value;
			}
		}
		cout << "读取Image数据完毕......" << endl;
	}
	file.close();
	return DataMat;
}

Mat read_mnist_label(const string fileName) {
	int magic_number;
	int number_of_items;

	Mat LabelMat;

	ifstream file(fileName, ios::binary);
	if (file.is_open())
	{
		cout << "成功打开标签集 ... " << endl;

		file.read((char*)&magic_number, sizeof(magic_number));
		if (!file.good()) {
			cerr << "Error reading magic number from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&number_of_items, sizeof(number_of_items));
		if (!file.good()) {
			cerr << "Error reading number of items from " << fileName << endl;
			return Mat();
		}
		magic_number = reverseInt(magic_number);
		number_of_items = reverseInt(number_of_items);

		cout << "幻数（文件格式）:" << magic_number << "  ;标签总数:" << number_of_items << endl;

		cout << "开始读取Label数据......" << endl;
		//CV_32SC1代表32位有符号整型 通道数为1
		LabelMat = Mat::zeros(number_of_items, 1, CV_32SC1);
		for (int i = 0; i < number_of_items; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			if (!file.good()) {
				cerr << "Error reading label data from " << fileName << " at row " << i << endl;
				return Mat();
			}
			LabelMat.at<int32_t>(i, 0) = static_cast<int32_t>(temp);
		}
		cout << "读取Label数据完毕......" << endl;

	}
	file.close();
	return LabelMat;
}
//将标签数据改为one-hot型
Mat one_hot(Mat label, int classes_num)
{
	int rows = label.rows;
	Mat one_hot = Mat::zeros(rows, classes_num, CV_32FC1);
	for (int i = 0; i < label.rows; i++)
	{
		int index = label.at<int32_t>(i, 0);
		if (index >= 0 && index < classes_num) {
			one_hot.at<float>(i, index) = 1.0;
		}
		else {
			cerr << "Invalid label index at row " << i << ": " << index << endl;
		}
	}
	return one_hot;
}

/*
* 画线的函数
*/
void linkOut2hid(cv::Mat& frame)
{
	cv::Point2i pt2;
	cv::Scalar color1(231, 148, 31);
	cv::Scalar color2(150, 150, 150);
	cv::Scalar color3(100, 100, 100);
	cv::Scalar color4(50, 50, 50);
	for (int i = 0; i < sizeof(g_outputLayerDraw) /sizeof(g_outputLayerDraw[0]); i++)
	{
		cv::Point2i pt1 = cv::Point2i(g_outputLayerDraw[i].x + g_gradw/2, g_outputLayerDraw[i].y+ g_gradh);//*10
		
		for (int j = 0; j < sizeof(g_hidLayerDraw) / sizeof(g_hidLayerDraw[0]); j++)
		{
			pt2 = cv::Point2i(g_hidLayerDraw[j].x+ 7.0/2, g_hidLayerDraw[j].y); //*64
			if (g_outputLayerDraw[i].v > 0.1 && g_hidLayerDraw[j].v > 0.3 )
			{
				cv::line(frame, pt1, pt2, color1, 1, 1);
			}
		}	
	}
}
//输入层到隐藏层的连线
void linkInput2Hid(cv::Mat& frame)
{
	cv::Point2i pt2;
	int cr = 0;
	
	for (int i = 0; i < sizeof(g_inputLayerDraw784) /sizeof(g_inputLayerDraw784[0]); i++)
	{
		cv::Point2i pt1 = cv::Point2i(g_inputLayerDraw784[i].x + 7/2, g_inputLayerDraw784[i].y);//*10
		
		for (int j = 0; j < sizeof(g_hidLayerDraw) / sizeof(g_hidLayerDraw[0]); j++)
		{
			pt2 = cv::Point2i(g_hidLayerDraw[j].x+ 7.0/2, g_hidLayerDraw[j].y +26); //*64
			if (g_inputLayerDraw784[i].v > 1 && g_hidLayerDraw[j].v > 0.7 )
			{
				//cout << "color = " << cr << " ";
				//cv::line(frame, pt1, pt2, cv::Scalar(173, 190, 241), 1, 1);
				cvui:line(frame, pt1, pt2, cv::Scalar(231, 148, 31),1,8,0);
				//cvui::rect(frame, pt1.x, pt1.y, rectangleR.width, rectangleR.height, 0xaaaaaa, 0xdaaaa0000);
			}
		}	
	}
}

//计算e为底的指数函数
void myExp(Mat& src, Mat& des)
{
	for (int i = 0; i < src.cols; i++)
	{
		des.at<double>(0, i) = exp(src.at<double>(0, i));
	}
}
//预测函数

//鼠标按键移动处理函数
void mouseAction(cv::Mat &frame)
{
	cv::Rect rectangleL(130, 10, 20, 20);
	cv::Rect rectangleR(150, 10, 20, 20);
	cvui::rect(frame, rectangleL.x, rectangleL.y, rectangleL.width, rectangleL.height, 0xaaaaaa, 0xdff000000);
	cvui::rect(frame, rectangleR.x, rectangleR.y, rectangleR.width, rectangleR.height, 0xaaaaaa, 0xdff000000);

	g_mouse_x = cvui::mouse().x;
	g_mouse_y = cvui::mouse().y;
	cvui::printf(frame, 10, 10, "(%d,%d)", cvui::mouse().x, cvui::mouse().y);
	// Did any mouse button go down? 按下的时刻调用一次
	if (cvui::mouse(cvui::DOWN)) {
		// Position the rectangle at the mouse pointer.
		//cvui::text(frame, 10, 70, "<-");
	}

	// Is any mouse button down (pressed)? //按下之后一直回调，适合按下之后鼠标书写
	if (cvui::mouse(cvui::IS_DOWN)) {
		// Adjust rectangle dimensions according to mouse pointer
	}

	// Did any mouse button go up?
	if (cvui::mouse(cvui::UP)) {
		// Hide the rectangle
	}

	// Was the mouse clicked (any button went down then up)?
	if (cvui::mouse(cvui::CLICK)) {
		//cvui::text(frame, 10, 70, " clicked!");
	}
	if (cvui::mouse(WINDOW_NAME, cvui::LEFT_BUTTON, cvui::IS_DOWN))
	{
		//cvui::text(frame, 10, 70, "<-");
		g_mouseKeyleftRight = 255;
		cvui::rect(frame, rectangleL.x, rectangleL.y, rectangleL.width, rectangleL.height, bdColor, filColor);
	}
	if (cvui::mouse(WINDOW_NAME, cvui::RIGHT_BUTTON, cvui::IS_DOWN))
	{
		//cvui::text(frame, 10, 70, "->");
		g_mouseKeyleftRight = 0;
		cvui::rect(frame, rectangleR.x, rectangleR.y, rectangleR.width, rectangleR.height, 0xaaaaaa, 0xdaaaa0000);
	}
}

//绘制手写板
void inputLayerDraw(cv::Mat& frame)
{
	//g_inputLayerDraw
	for (int j = 0; j < 28; j++)
		for (int i = 0; i < 28; i++)
		{
			if (g_inputLayerDraw[j * 28 + i].v == 255) {
				cvui::rect(frame, g_inRect.x + i * g_inRect.width, g_inRect.y + g_inRect.height * j, g_inRect.width, g_inRect.height, bdColor, filColor); //手写
			}
			else {
				cvui::rect(frame, g_inRect.x + i * g_inRect.width, g_inRect.y + g_inRect.height * j, g_inRect.width, g_inRect.height, 0xaaaaaa, 0xffa0a0a0);//画背景
			}
		}
	cout << "thread main ----------->" << endl;
}

//绘制手写展开区
void inputExpansionLayerDraw(cv::Mat& frame) //
{
	int count = 0;
	int gradwh = 7;
	for (int j = 0; j < 7; j++)
		for (int i = 0; i < 120; i++,count++)
		{
			//cvui::rect(frame, 80 + i * (gradwh+2),  360 + (gradwh+2) * j, gradwh, gradwh, 0xaaaaaa, 0xffa0a0a0);
			if (g_inputLayerDraw[count].v == 255) {
				
				//0xaa0000  0x00880000手写笔记颜色，数越小，颜色越深 //acolor
				cvui::rect(frame, 80 + i * (gradwh + 2), 394 + (gradwh + 4) * j , gradwh, gradwh, bdColor, filColor);
			}
			else {
				cvui::rect(frame, 80 + i * (gradwh + 2), 394 + (gradwh + 4) * j, gradwh, gradwh, 0xaaaaaa, 0xffa0a0a0);//画背景
			}
			g_inputLayerDraw784[count].x = 80 + i * (gradwh + 2);
			g_inputLayerDraw784[count].y = 394 + (gradwh + 4) * j;
			//g_inputLayerDraw784[count].v = g_inputLayerDraw[count].v;
			if (count >= 784 -1)break;
		}
}
//绘制隐藏层 64
void hidLayerDraw(cv::Mat& frame) //g_hidLayerDraw
{
	int gradwh = 14;

	for (int i = 0; i < 64; i++)
	{
		cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, gradwh + 8, 0xaaaaaa, 0xffa0a0a0);
		if (g_hidLayerDraw[i].v > 0) {
			int aa = g_hidLayerDraw[i].v * (gradwh + 8);
			aa = (aa == 0) ? 1 : aa;
			//cout << "aa= " << aa << endl;
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, aa, bdColor, filColor); //手写
		}
		else {
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, gradwh + 8, 0xaaaaaa, 0xffa0a0a0);//画背景
		}
		g_hidLayerDraw[i].x = 120 + i * (gradwh + 2);
		g_hidLayerDraw[i].y = 270;
	}
	//画权重的线段
	int i;
	cv::Point2i pt1;
	cv::Point2i pt2;

	for(int j=0;j < 10;j++)
	{
		for ( i = 0; i < 65; i++)
		{
			tempW = g_hidLayerWeights[i][j].v;
			float tempW_old = g_hidLayerWeights[i][j].vOld;
			if (abs(tempW - tempW_old) >= filter_step)//设置权重刷新的阈值filter_step
			{
				g_thickness = 3;
			}
			else 
			{
				g_thickness = 1;
			}
			g_hidLayerWeights[i][j].vOld = tempW;
			
			if (tempW >= 0)//判断正负，画不同颜色的线段
			{
				pt1 = cv::Point2i(120 + i * (gradwh + 2), 270 - (j * 10) - 5);
				pt2 = cv::Point2i(pt1.x  + gradwh * tempW, pt1.y);
				cv::line(frame, pt1, pt2, cv::Scalar(231, 148, 31), g_thickness, 8, 0);
			}
			else 
			{
				pt1 = cv::Point2i(120 + i * (gradwh + 2), 270 - (j * 10) - 5);
				pt2 = cv::Point2i(pt1.x + gradwh * (-1.0* tempW), pt1.y);
				cv::line(frame, pt1, pt2, cv::Scalar(48, 31,231), g_thickness, 8, 0);
			}
			//cout << "pt1: " << pt1 << " pt2: " << pt2 << endl;
		}
	}
}
//输出层
void outputLayerDraw(cv::Mat& frame)
{
	//int gradwh = 30;
	int x = 313, y = 108;
	int x2 = 191, y2 = 300;
	cv::Point2i pt1, pt2,pt3,pt4;
	cv::Scalar color1, color2;
	for (int i = 0; i < 10; i++)
	{
		cvui::rect(frame, x + i * (g_gradw + 15), y , g_gradw, g_gradh, 0xaaaaaa, 0xffa0a0a0);
		cvui::text(frame, x + i * (g_gradw + 15)+5, y-20, to_string(i), 0.7, 1);
		g_outputLayerDraw[i].x = x + i * (g_gradw + 15);
		g_outputLayerDraw[i].y = y;
		pt1 = cv::Point2i(x2, y2 + (8 * i + 8));
		pt2 = cv::Point2i( x2 + 100 * (2 - g_each_number_error[i]), y2 + (8 * i + 8));
		color1 = cv::Scalar(132, 135, 240);
		color2 = cv::Scalar(246, 130, 50);
		pt3 = cv::Point2i(x2+420, y2 + (8 * i + 8));
		pt4 = cv::Point2i(x2 +420 -100 * (2 - g_each_number_error2[i]), y2 + (8 * i + 8));
		//画出表示每个数字误差的矩形 gdata[k] _df.at<double>(0, d)
		//cvui::rect(frame, x2  , y2 +(8*i +8), 100 * (2-g_each_number_error[i]), 4, 0xaaaaaa, 0xffa0a0a0);
		//cvui::rect(frame, x2 +400, y2 + (8 * i + 8), 100 * (2 - g_each_number_error2[i]), 4, 0xaaaaaa, 0xffa0a0a0);
		cv::line(frame, pt1, pt2, color1, 2, 8, 0);
		cv::line(frame, pt3, pt4, color2, 2, 8, 0);
	}
}
int main(int argc, const char* argv[])
{
	cout << "@ 2025 DongHai XianRen\n控制台，运行时请勿关闭" << endl;
	bool Filter_0_001 = false;
	bool Filter_0_0001 = true;
	bool Filter_0_00001 = false;
	
	bool windowsShow = true;

	TrainThread trainthread("ThreadTrain");

	//outputs = Mat(1, 10, CV_32F, buf + 1568);
	outputs.create(1, 10, CV_64F);

	//for train
	 
	 //min_val = max_val = min_val1 = max_val1 = 0.;
	 max_val = 0.95; min_val = -max_val;
	 max_val1 = 0.98; min_val1 = -max_val1;

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //只打印错误信息
	// Create a frame where components will be rendered to.
	cv::Mat frame = cv::Mat(720, 1280, CV_8UC3);
	memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);


	while (windowsShow) {
		//global color 
		unsigned int r, g, b,a;// = (unsigned int)g_inputLayerDraw784[count].v;
		r = doubleColorValue_B;
		g = doubleColorValue_G << 8;
		b = doubleColorValue_R << 16;
		a = doubleColorValue_A << 24;
		bdColor = r | g | b | a;
		filColor = bdColor;
		//filColor |= 0xff00000000;

		// Fill the frame with a nice color
		frame = cv::Scalar(255, 255, 255);

		// Render UI components to the frame
		//cvui::text(frame, 226, 13, "Back Propagation Visualization可视化",1,1);
		cvZH::putTextZH(frame,"机器学习反向传播可视化",	cv::Point(226, 13),	CV_RGB(0, 0, 0),30);
		//cvui::text(frame, 110, 120, "cvui is awesome!",1,1);
		//int x = ;
		
		cvui::text(frame, 890, 500, "A", 0.6, 1);
		cvui::trackbar(frame, 895, 485, 150, &doubleColorValue_A, (uchar)0, (uchar)255, 0, "%.0Lf");
		cvui::text(frame, 890, 540, "R", 0.6, 1);
		cvui::trackbar(frame, 895, 525, 150, &doubleColorValue_R, (uchar)0, (uchar)255, 0, "%.0Lf");
		cvui::text(frame, 890, 580, "G", 0.6, 1);
		cvui::trackbar(frame, 895, 565, 150, &doubleColorValue_G, (uchar)0, (uchar)255, 0, "%.0Lf");
		cvui::text(frame, 890, 620, "B", 0.6, 1);
		cvui::trackbar(frame, 895, 605, 150, &doubleColorValue_B, (uchar)0, (uchar)255, 0, "%.0Lf");
		// g_train_sleep time
		cvui::text(frame, 608, 580, "Speed", 0.5, 1);
		unsigned int options = cvui::TRACKBAR_DISCRETE | cvui::TRACKBAR_HIDE_SEGMENT_LABELS;
		cvui::trackbar(frame, 671, 565, 150, &g_train_sleep_time, (int)0, (int)2000, 20, "%.0Lf", options, (int)10);

		//鼠标的处理
		mouseAction(frame);
		inputLayerDraw(frame);
		inputExpansionLayerDraw(frame);
		hidLayerDraw(frame);
		outputLayerDraw(frame);

		//显示预测结果
		cvui::text(frame, 777, 87, g_ResultString, 0.7, 1);
		
		//画出输出框的比例
		for (int i = 0; i < 10; i++)
		{
			//cout << i << "= " << g_outputLayerDraw[i].v << endl;
			//cout << g_outputLayerDraw[i].x << " " << g_outputLayerDraw[i].y << " " << g_gradh * g_outputLayerDraw[i].v + 1 << endl;
			cvui::rect(frame, g_outputLayerDraw[i].x, g_outputLayerDraw[i].y, g_gradw, g_gradh * g_outputLayerDraw[i].v + 1, bdColor, filColor); //手写
		}

		//button Start 事件处理
		if (cvui::button(frame, 666, 648, "Start")) {
			memset(g_inputLayerDraw, 0, sizeof(g_inputLayerDraw));
			memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));
			memset(g_hidLayerDraw, 0, sizeof(g_hidLayerDraw));
			memset(g_inputLayerDraw784, 0, sizeof(g_inputLayerDraw784));
			g_ResultString = "";

			//开始训练
			//startTrain();
			
			trainthread.Start();//线程开始调度，调用startTrain()
			//canmessageThread canthread("canthread");
			//canthread.Start();
		}
		if (cvui::button(frame, 1124, 8, "&Quit")) {
			break;
		}
		//if (cvui::button(frame, 57, 166, "Link")) {
		//	linkOut2hid(frame);
		//}
		//复选框
		cvui::text(frame, 36, 153, "Filter:", 0.4, 1);
		cvui::checkbox(frame, 37, 176, "0.001", &Filter_0_001);
		if (Filter_0_001) { Filter_0_0001 = Filter_0_00001 = false; filter_step = 0.001;};
		cvui::checkbox(frame, 37, 206, "0.0001", &Filter_0_0001);
		if (Filter_0_0001) { Filter_0_001 = Filter_0_00001 = false; filter_step = 0.0001; }
		cvui::checkbox(frame, 37, 236, "0.00001", &Filter_0_00001);
		if (Filter_0_00001) { Filter_0_001 = Filter_0_0001 = false; filter_step = 0.00001; }
		// Update cvui stuff and show everything on the screen
		//copyrighte of
		cvui::text(frame, 1004, 677, "@ 2025 DongHai XianRen", 0.4, 1);
		//cvui::text(frame, 41, 494, "MNIST Data set input ", 0.5, 1);
		//cvui::text(frame, 41, 304, "derivative 10 *1", 0.5, 1);
		 cvZH::putTextZH(frame, "偏差", cv::Point(41, 304), CV_RGB(0, 0, 0), 20);
		//cvui::text(frame, 132, 105, "Label index", 0.5, 1);
		cvZH::putTextZH(frame, "数字标签", cv::Point(132, 105), CV_RGB(0, 0, 0), 20);
		cvui::text(frame, 942, 105, "Weight update 65*10", 0.5, 1);
		//训练图片个数，
		//cvui::text(frame, 51, 526, "Trained pic: " + to_string(g_train_cycle) + "/60000",0.5,1);
		cvZH::putTextZH(frame, "训练图片个数/总数" , cv::Point(51, 526), CV_RGB(0, 0, 0), 20);
		cvui::text(frame, 240, 533,  to_string(g_train_cycle) + "/60000*"+to_string(g_train_count), 0.5, 1);
		//cvui::text(frame, 51, 546, "  Deviation: " + to_string(g_ErrorDeviation) , 0.5, 1);
		cvZH::putTextZH(frame, "偏差", cv::Point(51, 546), CV_RGB(0, 0, 0), 20);
		cvui::text(frame, 118, 553,  to_string(g_ErrorDeviation), 0.5, 1);

		cvZH::putTextZH(frame, "训练轮数", cv::Point(58, 501), CV_RGB(0, 0, 0), 20);
		cvui::counter(frame, 229, 501, &g_train_count);
		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();
		cvui::imshow(WINDOW_NAME, frame);

		int keyvalue = cv::waitKey(100);
		if (keyvalue == 27 || keyvalue == 81) { //ESC Q 按键退出
			break;
		} 
		if (keyvalue == 67) { //C 清除
			memset(g_inputLayerDraw, 0, sizeof(g_inputLayerDraw));
			memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));
			memset(g_hidLayerDraw, 0, sizeof(g_hidLayerDraw));
			memset(g_inputLayerDraw784, 0, sizeof(g_inputLayerDraw784));
			g_ResultString = "";
		}
		if (getWindowProperty(WINDOW_NAME, WND_PROP_AUTOSIZE) != 1)
		{
			break;
		}	
	}
	destroyAllWindows();
	return 0;
}