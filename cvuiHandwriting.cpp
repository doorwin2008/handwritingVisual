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
cv::Mat sigmoid(const cv::Mat& input) {
	cv::Mat output = input.clone();
	for (int i = 0; i < output.rows; ++i) {
		for (int j = 0; j < output.cols; ++j) {
			// 对于 CV_64F 类型数据
			if (output.type() == CV_64F) {
				double val = output.at<double>(i, j);
				output.at<double>(i, j) = 1.0 / (1.0 + std::exp(-val* 2.6));
			}
			// 对于 CV_32F 类型数据
			else if (output.type() == CV_32F) {
				float val = output.at<float>(i, j);
				output.at<float>(i, j) = 1.0f / (1.0f + std::exp(-val * 2.6));
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
	cv::Scalar color(200, 200, 200);
	for (int i = 0; i < sizeof(g_outputLayerDraw) /sizeof(g_outputLayerDraw[0]); i++)
	{
		cv::Point2i pt1 = cv::Point2i(g_outputLayerDraw[i].x + g_gradw/2, g_outputLayerDraw[i].y+ g_gradh);//*10
		
		for (int j = 0; j < sizeof(g_hidLayerDraw) / sizeof(g_hidLayerDraw[0]); j++)
		{
			pt2 = cv::Point2i(g_hidLayerDraw[j].x+ 7.0/2, g_hidLayerDraw[j].y); //*64
			if (g_outputLayerDraw[i].v > 0.5 && g_hidLayerDraw[j].v > 0.7 )
			{
				cv::line(frame, pt1, pt2, color, 1, 1);
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
			if (g_inputLayerDraw784[i].v ==255 && g_hidLayerDraw[j].v > 0.7 )
			{
				
			/*	if(g_hidLayerDraw[j].v >0.7 && g_hidLayerDraw[j].v <= 0.8 )cr = 50;
				else if (g_hidLayerDraw[j].v > 0.8 && g_hidLayerDraw[j].v <= 0.9)cr = 100;
				else if (g_hidLayerDraw[j].v > 0.9 && g_hidLayerDraw[j].v <= 1.0)cr = 150;
				else if (g_hidLayerDraw[j].v >  1.0)cr = 200;
				else cr = 0;
				cout << "color = " << cr << " ";*/
				cv::line(frame, pt1, pt2, cv::Scalar(173, 190, 241), 1, 1);
			}
		}	
	}
}
void PredicationStart(cv::Mat& frame)
{
	float inputArry[28 * 28]; 
	for (int i = 0; i < 28 * 28; i++)
	{
		inputArry[i] = g_inputLayerDraw[i].v;
	}
	Mat input = Mat(28, 28, CV_32FC1, inputArry);
	input.convertTo(input, CV_32F);
	//imput
	input = input / 255.0;
	//(1,784)
	input = input.reshape(1, 1);

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
	vector<Mat> layerOutputs;
	layerOutputs.push_back(input); // 输入层,把识别的图片矩阵导入一个矩阵向量

	// 获取每一层的权重矩阵
	vector<Mat> weights;
	Mat weightMat = ann->getWeights(0);
	//***** 第一层权重特殊处理。
	weights.push_back(weightMat);

	//从第二层开始，分别获取权重 前64和前10
	for (int i = 1; i < layerCount; ++i) {
		Mat weightMat = ann->getWeights(i);
		Mat w = weightMat.colRange(0, layer_sizes[i]);//***** 这里根据权重文件中过去的layer size读取实际的权重大小。
		weights.push_back(w);
	}

	//第一层的计算结果 date* w +b
	for (int jj = 0; jj < layerOutputs[0].cols; jj++) {
		double tt = layerOutputs[0].at<float>(0, jj);
		layerOutputs[0].at<float>(0, jj) = weights[0].at<double>(0, (2 * jj + 1)) + tt * weights[0].at<double>(0, (2 * jj));
	}

	// 第二层和第三层的计算 date* w +b
	for (int i = 1; i < layerCount; ++i) {
		Mat input = layerOutputs.back();//返回容器中最后一个元素
		//第二层和第三层的权重，包括最后一行的偏置值
		Mat weight = weights[i];
		if (weight.type() != CV_32F) {
			weight.convertTo(weight, CV_32F);
		}
		//输出结果
		Mat output;

		// 第二层和第三层的偏置
		Mat bias = weight.row(weight.rows - 1);//最后一行作为偏置
		//std::cout << "base of :" << i << std::endl;
		//for (int j = 0; j < bias.cols; ++j) {
			//std::cout << bias.at<float>(0, j) << " ";
		//}

		// 提取真正的权重，去掉最后一行
		Mat realWeight = weight.rowRange(0, weight.rows - 1);
		Mat temp = realWeight;
		output = input * temp;//+ bias

		layerOutputs.push_back(output);

		//临时调用激活函数，用来显示输出、
		Mat tempo = output.clone();
		tempo = sigmoid(tempo);
		// 打印当前层的权重和显示
		//std::cout << "weight of " << i << std::endl;
		for (int ii = 0; ii < tempo.rows; ++ii) {
			for (int jj = 0; jj < tempo.cols; ++jj) {
				//std::cout << tempo.at<float>(ii, jj) << " ";
				if (i == 1)
				{
					//std::cout << tempo.at<float>(ii, jj) << " ";
					g_hidLayerDraw[jj].v = tempo.at<float>(ii, jj);
				}
				
			}
		}
	}

	// 最后一层的输出即为预测结果
	Mat pre_out = layerOutputs.back();

	//Mat t = pre_out.t();
	//Mat output_one_row = t.reshape(1, 1);
	//激活函数，包括负值处理
	cv::Mat normalized_data = sigmoid2(pre_out);

	double maxVal = 0;
	double minVal = 0;
	cv::Point maxPoint;
	cv::Point minPoint;
	cv::minMaxLoc(normalized_data, &minVal, &maxVal, &minPoint, &maxPoint);
	int max_index = maxPoint.x;
	//cvui::printf(frame, 877, 97, "(Result: %d, Confidence: %d)", max_index, maxVal);
	//cout << "maxVal: "<<maxVal << endl;

	g_ResultString = "Result:" + to_string(max_index) + " Confidence:" + to_string(maxVal);

	if (pre_out.type() != CV_64F && pre_out.type() != CV_32F) {
		pre_out.convertTo(pre_out, CV_32F);
	}

	for (int i = 0; i < normalized_data.cols; i++)
	{
		g_outputLayerDraw[i].v  = normalized_data.at<float>(0, i);
	}
}
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
			if (gx > 1 && gx < 26 && gy >1 && gy < 26)
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
		cvui::rect(frame, rectangleL.x, rectangleL.y, rectangleL.width, rectangleL.height, 0xaaaaaa, 0xaaf00aa00);
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
				cvui::rect(frame, g_inRect.x + i * g_inRect.width, g_inRect.y + g_inRect.height * j, g_inRect.width, g_inRect.height, 0xaa0000, 0x0aa00000); //手写
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
				cvui::rect(frame, 80 + i * (gradwh + 2), 394 + (gradwh + 4) * j , gradwh, gradwh, 0xaa0000, 0x0aa00000); //手写
			}
			else {
				cvui::rect(frame, 80 + i * (gradwh + 2), 394 + (gradwh + 4) * j, gradwh, gradwh, 0xaaaaaa, 0xffa0a0a0);//画背景
			}
			g_inputLayerDraw784[count].x = 80 + i * (gradwh + 2);
			g_inputLayerDraw784[count].y = 394 + (gradwh + 4) * j;
			g_inputLayerDraw784[count].v = g_inputLayerDraw[count].v;
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
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, aa, 0xaa0000, 0x0aa00000); //手写
		}
		else {
			cvui::rect(frame, 120 + i * (gradwh + 2), 270, gradwh - 4, gradwh+8, 0xaaaaaa, 0xffa0a0a0);//画背景
		}
		g_hidLayerDraw[i].x = 120 + i * (gradwh + 2);
		g_hidLayerDraw[i].y = 270;
	}
}
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
	cout << "控制台，运行时请勿关闭" << endl;
	bool Display_output_link = true;
	bool Display_input_link = false;
	bool windowsShow = true;

	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //只打印错误信息
	// Create a frame where components will be rendered to.
	cv::Mat frame = cv::Mat(720, 1200, CV_8UC3);
	memset(g_outputLayerDraw, 0, sizeof(g_outputLayerDraw));

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	cvui::init(WINDOW_NAME);
	//加载ann模型
	 ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>("mnist_ann50.xml");

	while (windowsShow) {
		// Fill the frame with a nice color
		frame = cv::Scalar(255, 255, 255);

		// Render UI components to the frame
		cvui::text(frame, 226, 13, "ANN Handwriting Visualization",1,1);
		//cvui::text(frame, 110, 120, "cvui is awesome!",1,1);

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
			cvui::rect(frame, g_outputLayerDraw[i].x, g_outputLayerDraw[i].y, g_gradw, g_gradh * g_outputLayerDraw[i].v + 1, 0xaa0000, 0x0aa00000); //手写
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
		cvui::imshow(WINDOW_NAME, frame);

		// This function must be called *AFTER* all UI components. It does
		// all the behind the scenes magic to handle mouse clicks, etc.
		cvui::update();
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