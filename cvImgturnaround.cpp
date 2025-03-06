#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

// One (and only one) of your C++ files must define CVUI_IMPLEMENTATION
// before the inclusion of cvui.h to ensure its implementaiton is compiled.
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Hello World!"

int main()
{
    cv::Mat img = cv::imread("lena.jpg");

    //cv::Mat gray_img(img.cols, img.rows, CV_8UC1);
    //cv::cvtColor(img, gray_img, CV_BGR2GRAY);
    cv::Mat gray_img = img.clone();
    float theta = 45;
    float curvature = theta / 180 * CV_PI;

    cv::Mat x = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat y = cv::Mat::zeros(img.size(), CV_32FC1);

    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            // 计算新的坐标x.at取坐标的的像素值，round四舍五入取整数
            x.at<float>(row, col) = round(row * cos(curvature) - col * sin(curvature));
            y.at<float>(row, col) = round(row * sin(curvature) + col * cos(curvature));
        }
    }
    double x_min, x_max;
    double y_min, y_max;

    cv::minMaxLoc(x, &x_min, &x_max);
    cv::minMaxLoc(y, &y_min, &y_max);
    x = x - x_min;
    y = y - y_min;

    cv::minMaxLoc(x, &x_min, &x_max); // 画布的高
    cv::minMaxLoc(y, &y_min, &y_max); // 画布的宽

    cv::Mat dst = cv::Mat::zeros(x_max + 1, y_max + 1, CV_32FC1);  //幕布
    cv::Mat flag = cv::Mat::zeros(x_max + 1, y_max + 1, CV_8UC1);

    cout << "img type: "<<img.type() << endl;
    cout << "CV_32FC1: "<< CV_32FC1 << endl;
    cout << "CV_8UC1: " << CV_8UC1 << endl;
    cout << "sizeof ushort: " << sizeof(ushort) << endl;
    cout << "sizeof ushort: " << sizeof(ushort) << endl;

    for (int row = 0; row < gray_img.rows; row++)
    {
        for (int col = 0; col < gray_img.cols; col++)
        {
            int i = (int)x.at<float>(row, col);
            int j = (int)y.at<float>(row, col);
            dst.at<ushort>(i, j) = gray_img.at<ushort>(row, col);
            flag.at<ushort>(i, j) = 1;
        }
    }
    //均值插值法对空穴进行插值
    for (int row = 1; row < dst.rows - 1; row++)
    {
        for (int col = 1; col < dst.cols - 1; col++)
        {
            if (flag.at<ushort>(row, col - 1) == 1 && flag.at<ushort>(row, col + 1) == 1 &&
                flag.at<ushort>(row - 1, col) == 1 && flag.at<ushort>(row + 1, col) == 1
                && flag.at<ushort>(row, col) == 0)
            {
                dst.at<ushort>(row, col) = ushort((dst.at<ushort>(row, col - 1) + dst.at<ushort>(row, col + 1) +
                    dst.at<ushort>(row - 1, col) + dst.at<ushort>(row + 1, col)) / 4);
            }
        }
    }
    cv::imshow("input", gray_img);
    cv::imshow("output", dst);
    cv::waitKey(0);
    return 0;
}
