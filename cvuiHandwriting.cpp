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
* 权重文件，采用MINST训练的数据集
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

using namespace std;
using namespace cv;
using namespace cv::ml;

// One (and only one) of your C++ files must define CVUI_IMPLEMENTATION
// before the inclusion of cvui.h to ensure its implementaiton is compiled.
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Opencv ANN HandWrinting"
struct draw_s {
	int x;
	int y;
	float v;
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
//输出层坐标和权重信息，10个计算结果
draw_s g_outputLayerDraw[10];
int g_mouse_x;
int g_mouse_y;
cv::String g_ResultString = "";
int g_gradw = 30;
int g_gradh = 50;
cv::Ptr<cv::ml::ANN_MLP> ann;
uint8_t g_mouseKeyleftRight = 0;
//定义输出buffer
cv::AutoBuffer<double> _buf(1568 + 10);
//结果输出矩阵
Mat outputs;

// 获取每一层的权重矩阵,读出来是CV_64F
//Mat weightMat = ann->getWeights(0);
//***** 第一层权重特殊处理。
vector<Mat> weights;

cv::Mat sigmoid3(const cv::Mat& input) {
	cv::Mat output = input.clone();
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			// 对于 CV_64F 类型数据
			if (output.type() == CV_64F) {
				double val = output.at<double>(i, j);
				output.at<double>(i, j) = (1.0f - val) / (1.0f + val);
			}
			// 对于 CV_32F 类型数据
			else if (output.type() == CV_32F) {
				float val = output.at<float>(i, j);
		
				output.at<float>(i, j) = 1.0-std::exp(-val) / (1.0 + std::exp(  -val));
			}
		}
	}
	return output;
}
cv::Mat sigmoid2(const cv::Mat& input) {
	cv::Mat output = input.clone();
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			// 对于 CV_64F 类型数据
			if (output.type() == CV_64F) {
				double val = output.at<double>(i, j);
				output.at<double>(i, j) = (1.0f - val) / (1.0f + val);
			}
			// 对于 CV_32F 类型数据
			else if (output.type() == CV_32F) {
				float val = output.at<float>(i, j);
				//output.at<float>(i, j) = (1.0f - val)/ (1.0f + val); // <activation_function>SIGMOID_SYM</activation_function>
				//double t = scale2 * (1. - data[j]) / (1. + data[j]);
				//val = ((val < 0)? -1:val);
				output.at<float>(i, j) = 1.0 / (1.0 + std::exp(0.1 * -val));
			}
		}
	}
	return output;
}
cv::Mat activefunction(const cv::Mat& input) {
	cv::Mat output = input.clone();
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			// 对于 CV_64F 类型数据
			if (output.type() == CV_64F) {
				double val = output.at<double>(i, j);
				if (!cvIsInf(val))//如果遇到无穷小，值为 -1
				{
					output.at<double>(i, j) = (1. - val) / (1. + val);
				}
				else
				{
					output.at<double>(i, j) = -1;
				}
				
			}
			// 对于 CV_32F 类型数据
			else if (output.type() == CV_32F) {
				float val = output.at<float>(i, j);
				if (!cvIsInf(val))//如果遇到无穷小，值为 -1
				{
					output.at<float>(i, j) = (1. - val) / (1. + val);
				}
				else
				{
					output.at<float>(i, j) = -1;
				}
			}
		}
	}
	return output;
}
/*
* 画线的函数
*/
void linkOut2hid(cv::Mat& frame)
{
	cv::Point2i pt2;
	cv::Scalar color1(200, 200, 200);
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
				/*if (g_outputLayerDraw[i].v > 0.9 && g_hidLayerDraw[j].v > 0.5)
				{
					cv::line(frame, pt1, pt2, color4, 1, 1);
				}
				else if (g_outputLayerDraw[i].v > 0.6 && g_hidLayerDraw[j].v > 0.5)
				{
					cv::line(frame, pt1, pt2, color2, 1, 1);
				}
				else if (g_outputLayerDraw[i].v > 0.4 && g_hidLayerDraw[j].v > 0.3)
				{
					cv::line(frame, pt1, pt2, color3, 1, 1);
				}
				else if (g_outputLayerDraw[i].v > 0.1 && g_hidLayerDraw[j].v > 0.1)
				{
					cv::line(frame, pt1, pt2, color4, 1, 1);
				}*/
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
				
			/*	if(g_hidLayerDraw[j].v >0.7 && g_hidLayerDraw[j].v <= 0.8 )cr = 50;
				else if (g_hidLayerDraw[j].v > 0.8 && g_hidLayerDraw[j].v <= 0.9)cr = 100;
				else if (g_hidLayerDraw[j].v > 0.9 && g_hidLayerDraw[j].v <= 1.0)cr = 150;
				else if (g_hidLayerDraw[j].v >  1.0)cr = 200;
				else cr = 0;
				cout << "color = " << cr << " ";*/
				//cv::line(frame, pt1, pt2, cv::Scalar(173, 190, 241), 1, 1);
				cvui:line(frame, pt1, pt2, cv::Scalar(173, 190, 241),4,8,0);
				//cvui::rect(frame, pt1.x, pt1.y, rectangleR.width, rectangleR.height, 0xaaaaaa, 0xdaaaa0000);
			}
		}	
	}
}
void calc_activ_func(Mat& sums, const Mat& w)
{
	const double* bias = w.ptr<double>(w.rows - 1);
	int i, j, n = sums.rows, cols = sums.cols;
	double f_param2 = 1.0;
	double f_param1 = 1.0;
	double scale = 0, scale2 = f_param2;
	scale = -f_param1;
	
	CV_Assert(sums.isContinuous());

	for (i = 0; i < n; i++)
	{
		double* data = sums.ptr<double>(i);
		for (j = 0; j < cols; j++)
		{
			data[j] = (data[j] + bias[j]) * scale;
		}
	}

	exp(sums, sums);

	if (sums.isContinuous())
	{
		cols *= n;
		n = 1;
	}

	for (i = 0; i < n; i++)
	{
		double* data = sums.ptr<double>(i);
		for (j = 0; j < cols; j++)
		{
			if (!cvIsInf(data[j]))
			{
				double t = scale2 * (1. - data[j]) / (1. + data[j]);
				data[j] = t;
			}
			else
			{
				data[j] = -scale2;
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
void PredicationStart(cv::Mat& frame)
{
	//给输入
	float inputArry[28 * 28]; 
	for (int i = 0; i < 28 * 28; i++)
	{
		inputArry[i] = g_inputLayerDraw[i].v;
	}

	//格式化输入
	Mat inputs = Mat(28, 28, CV_32FC1, inputArry);
	inputs.convertTo(inputs, CV_64F);
	inputs = inputs / 255.0;
	//(1,784)
	inputs = inputs.reshape(1, 1); // 288*288 -> 1*784

	double* buf = _buf.data();
	
	//  outputs = outputs.getMat();
	//读取权重文件，转移到main函数中，因为只需要加载一次

	//float ret = ann->predict(input, pre_out);
	// 获取模型的层数
	//权重的计算，从第二层开始，bias偏移量保存在最后一行
	
	//获取网络层次结构信息 3行1列
	cv::Mat layerSizes = ann->getLayerSizes();

	//每层网络的大小的矩阵 存储到容器方便调用 784- 64 -10
	vector<int> layer_sizes;
	for (int i = 0; i < layerSizes.rows; ++i) {
		for (int j = 0; j < layerSizes.cols; ++j) {
			//std::cout << layerSizes.at<int>(i, j) << " ";
			layer_sizes.push_back(layerSizes.at<int>(i, j));
		}
	}
	//网络层数 3
	int layerCount = layerSizes.rows;

	// 每一层数据计算结果
	//vector<Mat> layerOutputs;
	//layerOutputs.push_back(input); // 输入层,把识别的图片矩阵导入一个矩阵向量

	Mat layer_in = inputs.rowRange(0, 1);
	Mat layer_out(1, layer_in.cols, CV_64F, buf);

	//第一层的计算结果 scale_input date* w +b
	//std::cout << endl << "inputs " << endl;
	for (int j = 0; j < inputs.cols; j++) {
		double _input = inputs.at<double>(0, j);
		inputs.at<double>(0, j) = weights[0].at<double>(0, (2 * j + 1)) + _input * weights[0].at<double>(0, (2 * j));
		//if ((float)(inputs.at<double>(0, j)) > 0)
		//	std::cout << (float)(inputs.at<double>(0, j)) << " ";
		g_inputLayerDraw784[j].v = (float)(inputs.at<double>(0, j));
	}

	// 第二层和第三层的计算 date* w +b
	for (int i = 1; i < layerCount; ++i) {
		double* data = buf + ((0 & 1) ? 784 * 1 : 0);
		int cols = layer_sizes[i];

		layer_out = Mat(1, cols, CV_64F, data);
		Mat weight = weights[i];
		if (weight.type() != CV_64F) { 
			weight.convertTo(weight, CV_64F);
		}
		// 第二层和第三层的偏置
		Mat bias = weight.row(weight.rows - 1);//最后一行作为偏置

		// 提取真正的权重，去掉最后一行
		//Mat realWeight = weight.rowRange(0, weight.rows - 1);
		Mat w = weight.rowRange(0, layer_in.cols);
		int a = w.type();
		int b = bias.type();
		int c = layer_in.type();
		
		//手动计算矩阵相乘
		layer_out = (layer_in * w);
		//或者使用cv库接口实现矩阵相乘
		//cv::gemm(layer_in, w, 1, cv::noArray(), 0, layer_out);//cv::gemm()实现矩阵相乘,优化了算法

		//激活函数	
		//calc_activ_func(layer_out, weights[i]);
		//或者使用手搓接口
		layer_out =(layer_out + bias)*(-1); 
		//myExp(layer_out, layer_out);//归一化，全部转为正数， 自然常数e为底的指数函数
		//或者调用opencv的库，直接计算两个矩阵元素的e为底的指数
		cv::exp(layer_out, layer_out);//自然常数e为底的指数函数
		layer_out = activefunction(layer_out);

		layer_in = layer_out;

		//临时调用激活函数，用来显示输出、
		Mat tempo = layer_out;

		//std::cout << "g_hidLayerDraw " << endl;
		for (int ii = 0; ii < tempo.rows; ++ii) {
			for (int jj = 0; jj < tempo.cols; ++jj) {
				if (i == 1)
				{
					//std::cout << (float)(tempo.at<double>(ii, jj)) << " ";
					g_hidLayerDraw[jj].v = (float)(tempo.at<double>(ii, jj));
				}
				else if (i == 2)//g_inputLayerDraw784
				{
					//std::cout << (float)(tempo.at<double>(ii, jj)) << " ";
					//g_hidLayerDraw[jj].v = (float)(tempo.at<double>(ii, jj));
				}
			}
		}
		layer_in = layer_out;
	}
	//scale_output
	layer_out = outputs.rowRange(0, 1);
	int cols = layer_out.cols;
	const double* w = weights[layerCount].ptr<double>();
	
	if (layer_out.type() == CV_64F)
	{
		for (int i = 0; i < layer_in.rows; i++)
		{
			const double* src = layer_in.ptr<double>(i);
			double* dst = layer_out.ptr<double>(i);
			for (int j = 0; j < 10; j++)
				dst[j] = (src[j] * w[j * 2] + w[j * 2 + 1]);
		}
	}
	// 最后一层的输出即为预测结果
	Mat pre_out = layer_out;

	//Mat t = pre_out.t();
	//Mat output_one_row = t.reshape(1, 1);
	//激活函数，包括负值处理
	
	double maxVal = 0;
	double minVal = 0;
	cv::Point maxPoint;
	cv::Point minPoint;
	cv::minMaxLoc(outputs, &minVal, &maxVal, &minPoint, &maxPoint);
	int max_index = maxPoint.x;

	g_ResultString = "Result:" + to_string(max_index) + " Confidence:" + to_string(maxVal);
	//cout << g_ResultString << endl;
	if (pre_out.type() != CV_64F) {
		pre_out.convertTo(pre_out, CV_64F);
	}

	for (int i = 0; i < pre_out.cols; i++)
	{
		g_outputLayerDraw[i].v  = (float)(pre_out.at<double>(0, i));
		if (g_outputLayerDraw[i].v < 0)g_outputLayerDraw[i].v = 0;
		//cout << "g_outputLayerDraw" << g_outputLayerDraw[i].v << endl;
	}
}
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
		//cvui::text(frame, 10, 70, " clicked!");
		if ((g_mouse_x > g_inRect.x && g_mouse_x < g_inRect.x + g_inRect.width*28)
			&& (g_mouse_y > g_inRect.y && g_mouse_y < g_inRect.y + g_inRect.height*28))
		{
			//cout << "g_mouse_x" << g_mouse_x << endl;

			int gx = (g_mouse_x - g_inRect.x) / g_inRect.width;
			int gy = (g_mouse_y - g_inRect.y) / g_inRect.height;
			g_inputLayerDraw[gy * 28 + gx].x = g_mouse_x;//绝对坐标用来画连接线
			g_inputLayerDraw[gy * 28 + gx].y = g_mouse_y;
			g_inputLayerDraw[gy * 28 + gx].v = g_mouseKeyleftRight; //左键书写，右键擦除
			if (gx > 1 && gx < 26 && gy >1 && gy < 26 && true) //如果需要加粗笔迹，置为true
			{
				int gxE = gx + 1;
				int gyE = gy + 1;
				g_inputLayerDraw[gyE * 28 + gxE].x = g_mouse_x;//绝对坐标用来画连接线
				g_inputLayerDraw[gyE * 28 + gxE].y = g_mouse_y;
				g_inputLayerDraw[gyE * 28 + gxE].v = g_mouseKeyleftRight; //左键书写，右键擦除
				gxE = gx ;
				gyE = gy + 1;
				g_inputLayerDraw[gyE * 28 + gxE].x = g_mouse_x;//绝对坐标用来画连接线
				g_inputLayerDraw[gyE * 28 + gxE].y = g_mouse_y;
				g_inputLayerDraw[gyE * 28 + gxE].v = g_mouseKeyleftRight; //左键书写，右键擦除
				gxE = gx + 1;
				gyE = gy ;
				g_inputLayerDraw[gyE * 28 + gxE].x = g_mouse_x;//绝对坐标用来画连接线
				g_inputLayerDraw[gyE * 28 + gxE].y = g_mouse_y;
				g_inputLayerDraw[gyE * 28 + gxE].v = g_mouseKeyleftRight; //左键书写，右键擦除
			}
		
			//目标识别开始,鼠标按下之后才开始识别
			PredicationStart(frame);
		}
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
		cvui::rect(frame, 120 + i * (gradwh + 2), 270 , gradwh-4, gradwh +8, 0xaaaaaa, 0xffa0a0a0);
		if (g_hidLayerDraw[i].v >0) {
			int aa = g_hidLayerDraw[i].v * (gradwh + 8);
			aa = (aa == 0) ? 1 : aa;
			//cout << "aa= " << aa << endl;
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, aa, bdColor, filColor); //手写
		}
		else {
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, gradwh+8, 0xaaaaaa, 0xffa0a0a0);//画背景
		}
		g_hidLayerDraw[i].x = 120 + i * (gradwh + 2);
		g_hidLayerDraw[i].y = 270;
	}
}
//输出层
void outputLayerDraw(cv::Mat& frame)
{
	//int gradwh = 30;
	int x = 313, y = 108;
	for (int i = 0; i < 10; i++)
	{
		cvui::rect(frame, x + i * (g_gradw + 15), y , g_gradw, g_gradh, 0xaaaaaa, 0xffa0a0a0);
		cvui::text(frame, x + i * (g_gradw + 15)+5, y-20, to_string(i), 0.7, 1);
		g_outputLayerDraw[i].x = x + i * (g_gradw + 15);
		g_outputLayerDraw[i].y = y;
	}
}
int main(int argc, const char* argv[])
{
	cout << "@ 2025 DongHai XianRen\n控制台，运行时请勿关闭" << endl;
	bool Display_output_link = true;
	bool Display_input_link = false;
	bool windowsShow = true;
	//outputs = Mat(1, 10, CV_32F, buf + 1568);
	outputs.create(1, 10, CV_64F);

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //只打印错误信息
	// Create a frame where components will be rendered to.
	cv::Mat frame = cv::Mat(720, 1200, CV_8UC3);
	memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);

	//加载ann模型
	 ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>("mnist_ann50.xml");
	 
	 //从第二层开始，分别获取权重 前64和前10
	 for (int i = 0; i < 3 + 2; ++i) {
		 Mat weightMat = ann->getWeights(i);
		 //Mat w = weightMat.colRange(0, layer_sizes[i]);//***** 这里根据权重文件中过去的layer size读取实际的权重大小。
		 weights.push_back(weightMat);
	 }
	 
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
		cvui::text(frame, 226, 13, "ANN Handwriting Visualization",1,1);
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

		//button处理
		if (cvui::button(frame, 666, 638, "&Clear")) {

			memset(g_inputLayerDraw, 0, sizeof(g_inputLayerDraw));
			memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));
			memset(g_hidLayerDraw, 0, sizeof(g_hidLayerDraw));
			memset(g_inputLayerDraw784, 0, sizeof(g_inputLayerDraw784));
			g_ResultString = "";
		}
		if (cvui::button(frame, 1124, 8, "&Quit")) {
			break;
		}
		//if (cvui::button(frame, 57, 166, "Link")) {
		//	linkOut2hid(frame);
		//}
		//复选框
		cvui::checkbox(frame, 37, 166, "Display output link", &Display_output_link);
		if(Display_output_link) linkOut2hid(frame);
		cvui::checkbox(frame, 37, 356, "Display input link", &Display_input_link);
		if (Display_input_link) linkInput2Hid(frame);
		// Update cvui stuff and show everything on the screen
		//copywrite of
		cvui::text(frame, 1004, 677, "@ 2025 DongHai XianRen", 0.4, 1);
		cvui::text(frame, 41, 494, "Input layer 28*28", 0.5, 1);
		cvui::text(frame, 41, 304, "Hid layer 64", 0.5, 1);
		cvui::text(frame, 132, 105, "outpt layer 10", 0.5, 1);
		

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();
		cvui::imshow(WINDOW_NAME, frame);
		if (cv::waitKey(20) == 27) {
			break;
		}
		if (getWindowProperty(WINDOW_NAME, WND_PROP_AUTOSIZE) != 1)
		{
			break;
		}	
	}
	destroyAllWindows();
	return 0;
}