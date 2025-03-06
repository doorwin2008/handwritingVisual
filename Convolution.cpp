#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int Convolution_1_channel(Mat& src, Mat& dst, Mat& kernel)
{
	if (kernel.rows != kernel.cols || kernel.rows % 2 == 0)
	{
		dst = src;
		cout << "kernel error!" << endl;
		return 0;
	}
	Mat srcB;
	int nrows = src.rows, ncols = src.cols, nk = kernel.rows, nb = (kernel.rows - 1) / 2;
	cout << kernel << endl;
	for (int i = 0; i < 3; i++)
	{
		float* p = kernel.ptr<float>(i);
		for(int j=0;j<3;j++) cout << p[j] << endl;
	}
	dst = src.clone();
	copyMakeBorder(src, srcB, nb, nb, nb, nb, BORDER_REFLECT);
	//imshow("border", srcB);
	for (int i = 0; i < nrows; i++)
	{
		uchar* pd = dst.ptr<uchar>(i);
		for (int j = 0; j < ncols; j++)
		{
			float value = 0;
			for (int ii = 0; ii < nk; ii++)
			{
				uchar* ps = srcB.ptr<uchar>(i + ii);
				float* pk = kernel.ptr<float>(ii);
				for (int jj = 0; jj < nk; jj++)
				{
					value += ps[j + jj] * pk[jj];
				}
			}
			//cout << value << endl;
			if (value < 0) value = 0;
			if (value > 255) value = 255;
			pd[j] = (int)value;
		}
	}
	return 1;
}

int Convolution(Mat& src, Mat& dst, Mat& kernel)
{
	vector<Mat> s_channels, d_channels;
	split(src, s_channels);
	for (int i = 0; i < s_channels.size(); i++)
	{
		Mat dst_1_channel;
		int f = Convolution_1_channel(s_channels.at(i), dst_1_channel, kernel);
		//if (i == 0) imshow("B_dst_1_channel", dst_1_channel);
		//if (i == 1) imshow("G_dst_1_channel", dst_1_channel);
		//if (i == 2) imshow("R_dst_1_channel", dst_1_channel);
		d_channels.push_back(dst_1_channel);
	}
	merge(d_channels, dst);
	return 1;
}

int main()
{
	Mat src = imread("D:/doorw_source/handwriting_opencv/dark.png");

	imshow("src", src);
	Mat edge;
	float a = 1.0 / 9;
	//Mat kernel1 = (Mat_<float>(3, 3) << a, a, a, a, a, a, a, a, a);
	Mat kernel2 = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
	//Convolution(src, src, kernel1);
	imshow("after fliter", src);//先平滑再提取边缘
	Convolution(src, edge, kernel2);
	Mat dst;
	add(src, edge, dst);
	imshow("edge", edge);
	//imwrite("edge.jpg", edge);
	imshow("dst", dst);
	//imwrite("dst.jpg", dst);
	waitKey(0);
	return 0;
}