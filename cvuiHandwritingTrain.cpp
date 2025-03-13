/*
* ��Ȩ���� �������˵� 2025��3�� 
* Bվ��https://space.bilibili.com/627167269
* ��������
* ��д�������빦�ܣ����ӻ�չʾ����ѧϰ������Ľ����ģ����Ԫ���ӣ���̬չʾʶ����̵ı仯
* ��Ʒ�����
* ������ʾ������cvui; Ȩ���ļ��Ķ�ȡ������opencv 4.10.0
* ����ṹ������� 28*28�����ز� 64 ������� 10
* ����� y=(1. - x) / (1. + x)
* ʹ��˵����
* Ȩ���ļ�������MINSTѵ�������ݼ�
*�����д���Ҽ�������esc�����˳�.���ĸ�ѡ�������ʾ��������Ԫ���ߡ�
*2025-3-8 ���¼�¼��λ��֮ǰ��32Ϊfloat�������¸�����С�����������жϳ������⣬�޸�Ϊ64λ��������㡣
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

#define WINDOW_NAME "CVUI Opencv ANN HandWrinting"
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
//��ɫ��
uchar doubleColorValue_R = 31;
uchar doubleColorValue_G = 131;
uchar doubleColorValue_B = 231;
uchar doubleColorValue_A = 128;
unsigned int  bdColor;
unsigned int  filColor;
//��д������
draw_s g_inputLayerDraw[28*28];
cv::Rect g_inRect = { 400,480,7,7 };
//չ��������ʾ
draw_s g_inputLayerDraw784[28 * 28];
//���ز������Ȩ����Ϣ��64��������
draw_s g_hidLayerDraw[64];
draw_sHistory g_hidLayerWeights[65][10];
//����������Ȩ����Ϣ��10��������
draw_s g_outputLayerDraw[10];
int g_mouse_x;
int g_mouse_y;
cv::String g_ResultString = "";
int g_gradw = 30;
int g_gradh = 50;

uint8_t g_mouseKeyleftRight = 0;
//�������buffer
cv::AutoBuffer<double> _buf(1568 + 10);
//����������
Mat outputs;

// for train 
cv::Ptr<cv::ml::ANN_MLP> ann;
//С�˴洢ת��
int reverseInt(int i);
//��ȡimage���ݼ���Ϣ
Mat read_mnist_image(const string fileName);
//��ȡlabel���ݼ���Ϣ
Mat read_mnist_label(const string fileName);
//����ǩ���ݸ�Ϊone-hot��
Mat one_hot(Mat label, int classes_num);
vector<int> layer_sizes= { 784,64,10 };;
double min_val, max_val, min_val1, max_val1;
vector<Mat> weights;
RNG rng;
int paramstermCrittype = 3;
bool trained;
int max_lsize=784;
int g_train_cycle = 0;
long g_ErrorDeviation = 0;
float g_each_number_error[10] ;
float g_each_number_error2[10];

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
class canmessageThread : public CThread
{
public:
	canmessageThread(const std::string& strName)
		: m_strThreadName2(strName)
	{
	}

	~canmessageThread()
	{
	}

public:
	virtual void Run()
	{
		cout << "can message transfer thread" << endl;
	}
private:
	std::string m_strThreadName2;
};

class TestThread : public CThread
{
public:
	TestThread(const std::string& strName)
		: m_strThreadName(strName)
	{
	}

	~TestThread()
	{
	}

public:
	virtual void Run()
	{
		/*CLock lock(g_metux);
		for (int i = 0; i < 10; i++)
		{
			CLock lock(g_metux);
			std::cout << m_strThreadName << ":" << i << std::endl;
			Sleep(100);
		}
		CLock unlock(g_metux);*/
		startTrain();
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
//������
//_xf��һ�� Mat ���͵����ã�ͨ����ʾ�������������ں���ִ�й����лᱻ�޸ġ�
//_df��һ�� Mat ���͵����ã����ڴ洢�����������ƫ�������ں���ִ�й����лᱻ�޸ġ�
//w��һ�� const Mat ���͵����ã�ͨ����ʾȨ�ؾ��󣬺���ִ�й����в����޸�����

void calc_activ_func_deriv(Mat& _xf, Mat& _df, const Mat& w)
{
	//��ȡƫ��ֵ��
	const double* bias = w.ptr<double>(w.rows - 1);
	//��ȡ������������ _xf ������ n ������ cols������ʼ�������������� scale �� scale2��
	int i, j, n = _xf.rows, cols = _xf.cols;
	{
		double scale = 1.0;
		double scale2 = 1.0;
		//��һ��ѭ���������������͵�������
		for (i = 0; i < n; i++)
		{
			double* xf = _xf.ptr<double>(i);
			double* df = _df.ptr<double>(i);

			for (j = 0; j < cols; j++)
			{
				//�������������� _xf ��ÿһ��Ԫ�ؼ��϶�Ӧ��ƫ��ֵ��Ȼ������������� scale
				xf[j] = (xf[j] + bias[j]) * scale;
				df[j] = -fabs(xf[j]);//�󸡵����� ����ֵ��
			}
		}
		// _df �����ÿһ��Ԫ�ؼ��� e ���ݴη�
		exp(_df, _df);

		// ((1+exp(-ax))^-1)'=a*((1+exp(-ax))^-2)*exp(-ax);
		// ((1-exp(-ax))/(1+exp(-ax)))'=(a*exp(-ax)*(1+exp(-ax)) + a*exp(-ax)*(1-exp(-ax)))/(1+exp(-ax))^2=
		// 2*a*exp(-ax)/(1+exp(-ax))^2
		scale *= 2 * 1.0;//������������

		//�ڶ���ѭ���������������͵�������
		//���㼤����ĵ��������� _df ����ͬʱ���� _xf �����ֵ��
		//���� xf[j] ������ȷ������ s0��Ȼ������м���� t0 �� t1��
		//������ _df �� _xf ����Ķ�ӦԪ��
		//������� y = (1 - e ^ x )/��1 + e ^ x)�ĵ�������Ϊ�� y' = �� -2e ^ x ��/(1 + e ^ x) ^ 2
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
/*�ú���ʵ����������ķ��򴫲�ѵ���㷨��ͨ�����ϵ���Ȩ������С����
���岽�����ǰ�򴫲�������������������򴫲�����Ȩ�أ�ֱ��������ֹ������
* ����˵��
const Mat& inputs���������ݾ���ÿһ�д���һ������������
const Mat& outputs������������ݾ���ÿһ�ж�Ӧһ���������������������
const Mat& _sw������Ȩ�ؾ������ڶԲ�ͬ�����������м�Ȩ��
TermCriteria termCrit��ѵ����ֹ���������������������������ֵ��
*/
int train_backprop(const Mat& inputs, const Mat& outputs, const Mat& _sw, TermCriteria termCrit)
{
	int i, j, k;
	//prev_E 8.9884656743115785e+307 �� E���ֱ�洢��һ�ε����͵�ǰ��������
	double prev_E = DBL_MAX * 0.5, E = 0; // DBL_MAX 1.7976931348623158e+308
	//itype �� otype���ֱ�洢��������������������͡�
	int itype = inputs.type(), otype = outputs.type();
	//count������������������
	int count = inputs.rows;

	//max_iter��������������
	int iter = -1, max_iter = termCrit.maxCount * count; 
	//epsilon =6�������ֵ��CV_TERMCRIT_EPS=2
	double epsilon = (termCrit.type & CV_TERMCRIT_EPS) ? termCrit.epsilon * count : 0;
	//l_count��������Ĳ�����
	int l_count = layer_count();
	//ivcount����������Ԫ������
	int ivcount = layer_sizes[0];
	//ovcount����������Ԫ������
	int ovcount = layer_sizes.back();

	// allocate buffers
	vector<vector<double> > x(l_count);
	vector<vector<double> > df(l_count);
	// �洢ÿһ��Ȩ�صĸ�����
	vector<Mat> dw(l_count);
	// ���仺����
	for (i = 0; i < l_count; i++)
	{
		int n = layer_sizes[i];
		x[i].resize(n + 1);//x���洢ÿһ��������
		df[i].resize(n);//df���洢ÿһ�㼤����ĵ�����
		dw[i] = Mat::zeros(weights[i].size(), CV_64F);//dw���洢ÿһ��Ȩ�صĸ�������
	}
	//��ʼ����������
	Mat _idx_m(1, count, CV_32S);
	int* _idx = _idx_m.ptr<int>();//_idx_m���洢���������������������������˳��
	for (i = 0; i < count; i++)
		_idx[i] = i;
	//��ʼ��������
	AutoBuffer<double> _buf(max_lsize * 2);//_buf�����ڴ洢��ʱ���ݡ�
	double* buf[] = { _buf.data(), _buf.data() + max_lsize };//sw������Ȩ�ؾ����ָ�롣

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
	//���򴫲�ѵ��ѭ�����ܵ�ͼƬ�ĸ�����60000 max_iter
	for (iter = 0; iter < max_iter; iter++)
	{
		//doorwin ����ÿ��ͼƬ��ԭʼ���ݣ�������д����ʾ
		memset(g_inputLayerDraw, 0, sizeof(g_inputLayerDraw));
		cout << "Thread Train==============> " << iter <<"/60000" << endl;
		g_train_cycle = iter;
		const uchar* ppp = inputs.ptr(iter);
		const float* ccc = (const float*)ppp;
		for (int m = 0; m < 784; m++)
		{
			if (ccc[m] > 0)g_inputLayerDraw[m].v = 255;
		}
		//���0-9��ˢ�£�ͨ���������ı�ǩ��Ϣ
		const uchar* ddd = outputs.ptr(iter);
		const float* eee = (const float*)ddd;
		memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));
		for (int n = 0; n < 10; n++)
		{
			if (eee[n] == 1)g_outputLayerDraw[n].v = 1;
		}
		Sleep(g_train_sleep_time); //���̹߳�����ʾҳ��ˢ����һ��
		//doorwin
		//����������
		int idx = iter % count; //idx����ǰ���������������
		double sweight = sw ? count * sw[idx] : 1.;//��ǰ������Ȩ�ء�

		//��������һ�����������󣬼������Ƿ�С����ֵ���������ֹͣѵ����ͬʱ�������������˳��
		if (idx == 0)
		{
			//printf("%d. E = %g\n", iter/count, E);
			if (fabs(prev_E - E) < epsilon)
				break;
			prev_E = E;
			E = 0;

			// shuffle indices �������
			for (i = 0; i < count; i++)
			{
				j = rng.uniform(0, count);
				k = rng.uniform(0, count);
				std::swap(_idx[j], _idx[k]);
			}
		}
		//ǰ�򴫲�
		idx = _idx[idx];

		const uchar* x0data_p = inputs.ptr(idx);
		const float* x0data_f = (const float*)x0data_p;
		const double* x0data_d = (const double*)x0data_p;

		double* w = weights[0].ptr<double>();
		//ÿ��ͼƬ������ؼ���
		for (j = 0; j < ivcount; j++)
		{
			x[0][j] = (itype == CV_32F ? (double)x0data_f[j] : x0data_d[j]) * w[j * 2] + w[j * 2 + 1];
		}	

		Mat x1(1, ivcount, CV_64F, &x[0][0]);

		//������㿪ʼ��������ÿһ������ x �ͼ�����ĵ��� df��
		// forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
		for (i = 1; i < l_count; i++)
		{
			int n = layer_sizes[i];
			Mat x2(1, n, CV_64F, &x[i][0]);
			Mat _w = weights[i].rowRange(0, x1.cols);
			gemm(x1, _w, 1, noArray(), 0, x2);//�����
			Mat _df(1, n, CV_64F, &df[i][0]);//df�洢ÿһ��Ȩ�صĸ�����
			calc_activ_func_deriv(x2, _df, weights[i]);//���������,ͨ���������㣬���ظ�����df�����뺯������x֮���������
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
		//. ������� 
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
		E *= sweight; //������������� grad1�����ۼ���� E��
		g_ErrorDeviation = E;
		//cout << endl << " E= " << E << endl;

		// ���򴫲� backward pass, update weights
		//������㿪ʼ��������Ȩ�� weights ����� grad1��
		//multiply ��������Ԫ�ؼ��˷���
		//add �������ھ���ӷ���
		for (i = l_count - 1; i > 0; i--)
		{
			int n1 = layer_sizes[i - 1], n2 = layer_sizes[i];
			Mat _df(1, n2, CV_64F, &df[i][0]);
			multiply(grad1, _df, grad1);//����Ԫ�صĳ˷����� 3 = 1*2
			Mat _x(n1 + 1, 1, CV_64F, &x[i - 1][0]);
			x[i - 1][n1] = 1.;
			//ǰ����������� 4 = 1*2*3
			gemm(_x, grad1,0.01 , dw[i], 0.1, dw[i]);////�����params.bpDWScale  params.bpMomentScale
			//ǰ����������� 3=1+2
			add(weights[i], dw[i], weights[i]);
			if (i > 1)
			{
				Mat grad2(1, n1, CV_64F, buf[i & 1]);
				Mat _w = weights[i].rowRange(0, n1);
				gemm(grad1, _w, 1, noArray(), 0, grad2, GEMM_2_T);//�����
				grad1 = grad2;
			}
		}
		//���weights��Ϣ��HMI��ʾ
		int tp = weights[2].type();
		for (int j = 0; j < 65; j++)
			for (int i = 0; i < 10; i++)
			{
				g_hidLayerWeights[j][i].v = (float)weights[2].at<double>(j, i);
			}
		//ͳ�����
		//cout<<"gdata" << endl;
		for (int i = 0; i < 10; i++)
		{
			g_each_number_error[i] = gdata[i];
			//cout << gdata[i] << "\t";
		}
		//cout  << endl;
	}//end for ����ͼƬ
	
	//���ص�������
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
	termcrit.maxCount = std::max((paramstermCrittype & CV_TERMCRIT_ITER ? 1 : MAX_ITER), 1);
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
	---------��һ���֣�ѵ������׼��-----------
	*/
	//��ȡѵ����ǩ���� (60000,1) ����Ϊint32
	Mat train_labels = read_mnist_label(train_labels_path);
	if (train_labels.empty()) {
		cerr << "Failed to read training labels." << endl;
		return -1;
	}
	//ann������ı�ǩ������ҪתΪone-hot�� �������ڽ���ɢ�ķ����ǩת��Ϊ������������
	train_labels = one_hot(train_labels, 10);

	//��ȡѵ��ͼ������ (60000,784) ����Ϊfloat32 ����δ��һ��
	Mat train_images = read_mnist_image(train_images_path);
	if (train_images.empty()) {
		cerr << "Failed to read training images." << endl;
		return -1;
	}
	//��ͼ�����ݹ�һ��
	train_images = train_images / 255.0;

	/*
	---------�ڶ����֣�����annѵ��ģ�Ͳ�����ѵ��-----------
	*/
	ann = cv::ml::ANN_MLP::create();
	//����ģ�͵Ĳ�νṹ �����Ϊ784 ���ز�Ϊ64 �����Ϊ10
	Mat layerSizes = (Mat_<int>(1, 3) << 784, 64, 10);
	ann->setLayerSizes(layerSizes);
	//���ò�������Ϊ���򴫲���
	ann->setTrainMethod(ANN_MLP::BACKPROP, 0.001, 0.1);
	//���ü����Ϊsigmoid
	ann->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	//���õ������� ���ѵ������Ϊ100
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 1, 0.0001));
	Mat ts = ann->getLayerSizes() ;
	for (int i = 0; i < ts.rows +2; i++)
	{
		weights.push_back( ann->getWeights(i));
	}
	//��ʼѵ��
	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_images, cv::ml::ROW_SAMPLE, train_labels);
	cout << "��ʼ����ѵ��..." << endl;
	if (!HWtrain(train_data,0)) {
		cerr << "Training failed." << endl;
		return -1;
	}
	cout << "ѵ�����" << endl;

	//����ģ��
	ann->save("mnist_ann.xml");

	cout << "�����˳�" << endl;
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
		cout << "�ɹ���ͼ�� ..." << endl;

		file.read((char*)&magic_number, sizeof(magic_number));//�������ļ���ʽ��
		if (!file.good()) {
			cerr << "Error reading magic number from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&number_of_images, sizeof(number_of_images));//ͼ������
		if (!file.good()) {
			cerr << "Error reading number of images from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&n_rows, sizeof(n_rows));//ÿ��ͼ�������
		if (!file.good()) {
			cerr << "Error reading number of rows from " << fileName << endl;
			return Mat();
		}
		file.read((char*)&n_cols, sizeof(n_cols));//ÿ��ͼ�������
		if (!file.good()) {
			cerr << "Error reading number of columns from " << fileName << endl;
			return Mat();
		}

		magic_number = reverseInt(magic_number);
		number_of_images = reverseInt(number_of_images);
		n_rows = reverseInt(n_rows);
		n_cols = reverseInt(n_cols);
		cout << "�������ļ���ʽ��:" << magic_number
			<< " ͼ������:" << number_of_images
			<< " ÿ��ͼ�������:" << n_rows
			<< " ÿ��ͼ�������:" << n_cols << endl;

		cout << "��ʼ��ȡImage����......" << endl;

		DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32FC1);
		for (int i = 0; i < number_of_images; i++) {
			for (int j = 0; j < n_rows * n_cols; j++) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				if (!file.good()) {
					cerr << "Error reading pixel data from " << fileName << " at row " << i << ", col " << j << endl;
					return Mat();
				}
				//������������һ����ÿ������ֵ��һ��
				float pixel_value = float(temp);
				//�����н�����ֵһ����д��Mat��
				DataMat.at<float>(i, j) = pixel_value;
			}
		}
		cout << "��ȡImage�������......" << endl;
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
		cout << "�ɹ��򿪱�ǩ�� ... " << endl;

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

		cout << "�������ļ���ʽ��:" << magic_number << "  ;��ǩ����:" << number_of_items << endl;

		cout << "��ʼ��ȡLabel����......" << endl;
		//CV_32SC1����32λ�з������� ͨ����Ϊ1
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
		cout << "��ȡLabel�������......" << endl;

	}
	file.close();
	return LabelMat;
}
//����ǩ���ݸ�Ϊone-hot��
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
* ���ߵĺ���
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
//����㵽���ز������
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

//����eΪ�׵�ָ������
void myExp(Mat& src, Mat& des)
{
	for (int i = 0; i < src.cols; i++)
	{
		des.at<double>(0, i) = exp(src.at<double>(0, i));
	}
}
//Ԥ�⺯��

//��갴���ƶ�������
void mouseAction(cv::Mat &frame)
{
	cv::Rect rectangleL(130, 10, 20, 20);
	cv::Rect rectangleR(150, 10, 20, 20);
	cvui::rect(frame, rectangleL.x, rectangleL.y, rectangleL.width, rectangleL.height, 0xaaaaaa, 0xdff000000);
	cvui::rect(frame, rectangleR.x, rectangleR.y, rectangleR.width, rectangleR.height, 0xaaaaaa, 0xdff000000);

	g_mouse_x = cvui::mouse().x;
	g_mouse_y = cvui::mouse().y;
	cvui::printf(frame, 10, 10, "(%d,%d)", cvui::mouse().x, cvui::mouse().y);
	// Did any mouse button go down? ���µ�ʱ�̵���һ��
	if (cvui::mouse(cvui::DOWN)) {
		// Position the rectangle at the mouse pointer.
		//cvui::text(frame, 10, 70, "<-");
	}

	// Is any mouse button down (pressed)? //����֮��һֱ�ص����ʺϰ���֮�������д
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

//������д��
void inputLayerDraw(cv::Mat& frame)
{
	//g_inputLayerDraw
	for (int j = 0; j < 28; j++)
		for (int i = 0; i < 28; i++)
		{
			if (g_inputLayerDraw[j * 28 + i].v == 255) {
				cvui::rect(frame, g_inRect.x + i * g_inRect.width, g_inRect.y + g_inRect.height * j, g_inRect.width, g_inRect.height, bdColor, filColor); //��д
			}
			else {
				cvui::rect(frame, g_inRect.x + i * g_inRect.width, g_inRect.y + g_inRect.height * j, g_inRect.width, g_inRect.height, 0xaaaaaa, 0xffa0a0a0);//������
			}
		}
	cout << "thread main ----------->" << endl;
}

//������дչ����
void inputExpansionLayerDraw(cv::Mat& frame) //
{
	int count = 0;
	int gradwh = 7;
	for (int j = 0; j < 7; j++)
		for (int i = 0; i < 120; i++,count++)
		{
			//cvui::rect(frame, 80 + i * (gradwh+2),  360 + (gradwh+2) * j, gradwh, gradwh, 0xaaaaaa, 0xffa0a0a0);
			if (g_inputLayerDraw[count].v == 255) {
				
				//0xaa0000  0x00880000��д�ʼ���ɫ����ԽС����ɫԽ�� //acolor
				cvui::rect(frame, 80 + i * (gradwh + 2), 394 + (gradwh + 4) * j , gradwh, gradwh, bdColor, filColor);
			}
			else {
				cvui::rect(frame, 80 + i * (gradwh + 2), 394 + (gradwh + 4) * j, gradwh, gradwh, 0xaaaaaa, 0xffa0a0a0);//������
			}
			g_inputLayerDraw784[count].x = 80 + i * (gradwh + 2);
			g_inputLayerDraw784[count].y = 394 + (gradwh + 4) * j;
			//g_inputLayerDraw784[count].v = g_inputLayerDraw[count].v;
			if (count >= 784 -1)break;
		}
}
//�������ز� 64
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
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, aa, bdColor, filColor); //��д
		}
		else {
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, gradwh + 8, 0xaaaaaa, 0xffa0a0a0);//������
		}
		g_hidLayerDraw[i].x = 120 + i * (gradwh + 2);
		g_hidLayerDraw[i].y = 270;
	}
	//��Ȩ�ص��߶�
	int i;
	cv::Point2i pt1;
	cv::Point2i pt2;

	for(int j=0;j < 10;j++)
	{
		for ( i = 0; i < 65; i++)
		{
			tempW = g_hidLayerWeights[i][j].v;
			float tempW_old = g_hidLayerWeights[i][j].vOld;
			if (abs(tempW - tempW_old) >= filter_step)//����Ȩ��ˢ�µ���ֵfilter_step
			{
				g_thickness = 3;
			}
			else 
			{
				g_thickness = 1;
			}
			g_hidLayerWeights[i][j].vOld = tempW;
			
			if (tempW >= 0)//�ж�����������ͬ��ɫ���߶�
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
//�����
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
		//������ʾÿ���������ľ��� gdata[k] _df.at<double>(0, d)
		//cvui::rect(frame, x2  , y2 +(8*i +8), 100 * (2-g_each_number_error[i]), 4, 0xaaaaaa, 0xffa0a0a0);
		//cvui::rect(frame, x2 +400, y2 + (8 * i + 8), 100 * (2 - g_each_number_error2[i]), 4, 0xaaaaaa, 0xffa0a0a0);
		cv::line(frame, pt1, pt2, color1, 2, 8, 0);
		cv::line(frame, pt3, pt4, color2, 2, 8, 0);
	}
}
int main(int argc, const char* argv[])
{
	cout << "@ 2025 DongHai XianRen\n����̨������ʱ����ر�" << endl;
	bool Filter_0_001 = false;
	bool Filter_0_0001 = true;
	bool Filter_0_00001 = false;
	
	bool windowsShow = true;

	TestThread thread2("Thread2");

	//outputs = Mat(1, 10, CV_32F, buf + 1568);
	outputs.create(1, 10, CV_64F);

	//for train
	 
	 //min_val = max_val = min_val1 = max_val1 = 0.;
	 max_val = 0.95; min_val = -max_val;
	 max_val1 = 0.98; min_val1 = -max_val1;

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //ֻ��ӡ������Ϣ
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
		cvui::text(frame, 226, 13, "Back Propagation Visualization",1,1);
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
		cvui::trackbar(frame, 671, 565, 150, &g_train_sleep_time, (int)50, (int)2000, 20, "%.0Lf", options, (int)10);

		//���Ĵ���
		mouseAction(frame);
		inputLayerDraw(frame);
		inputExpansionLayerDraw(frame);
		hidLayerDraw(frame);
		outputLayerDraw(frame);

		//��ʾԤ����
		cvui::text(frame, 777, 87, g_ResultString, 0.7, 1);
		
		//���������ı���
		for (int i = 0; i < 10; i++)
		{
			//cout << i << "= " << g_outputLayerDraw[i].v << endl;
			//cout << g_outputLayerDraw[i].x << " " << g_outputLayerDraw[i].y << " " << g_gradh * g_outputLayerDraw[i].v + 1 << endl;
			cvui::rect(frame, g_outputLayerDraw[i].x, g_outputLayerDraw[i].y, g_gradw, g_gradh * g_outputLayerDraw[i].v + 1, bdColor, filColor); //��д
		}

		//button����
		if (cvui::button(frame, 666, 648, "Start")) {
			memset(g_inputLayerDraw, 0, sizeof(g_inputLayerDraw));
			memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));
			memset(g_hidLayerDraw, 0, sizeof(g_hidLayerDraw));
			memset(g_inputLayerDraw784, 0, sizeof(g_inputLayerDraw784));
			g_ResultString = "";

			//��ʼѵ��
			//startTrain();
			
			thread2.Start();
			//canmessageThread canthread("canthread");
			//canthread.Start();
		}
		if (cvui::button(frame, 1124, 8, "&Quit")) {
			break;
		}
		//if (cvui::button(frame, 57, 166, "Link")) {
		//	linkOut2hid(frame);
		//}
		//��ѡ��
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
		cvui::text(frame, 41, 494, "MINST Data set input ", 0.5, 1);
		cvui::text(frame, 41, 304, "derivative 10 *1", 0.5, 1);
		cvui::text(frame, 132, 105, "Label index", 0.5, 1);
		cvui::text(frame, 942, 105, "Weight update 65*10", 0.5, 1);
		//ѵ��ͼƬ������
		cvui::text(frame, 51, 526, "Trained pic: " + to_string(g_train_cycle) + "/60000",0.5,1);
		cvui::text(frame, 51, 546, "  Deviation: " + to_string(g_ErrorDeviation) , 0.5, 1);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();
		cvui::imshow(WINDOW_NAME, frame);

		int keyvalue = cv::waitKey(100);
		if (keyvalue == 27 || keyvalue == 81) { //ESC Q �����˳�
			break;
		} 
		if (keyvalue == 67) { //C ���
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