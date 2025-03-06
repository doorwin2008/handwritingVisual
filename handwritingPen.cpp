#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cstring>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;
using namespace cv::ml;


// 函数用于将笔迹加宽并平滑
std::vector<std::pair<int, int>> widenAndSmoothStroke(const std::vector<std::pair<int, int>>& inputPoints) {
    // 创建一个 28x28 的空白图像
    cv::Mat image = cv::Mat::zeros(28, 28, CV_8UC1);

    // 将输入的坐标点绘制到图像上
    for (const auto& point : inputPoints) {
        int x = point.first;
        int y = point.second;
        if (x >= 0 && x < 28 && y >= 0 && y < 28) {
            image.at<uchar>(y, x) = 255;
        }
    }

    // 定义膨胀核
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // 进行形态学膨胀操作，加宽笔迹
    cv::Mat dilatedImage;
    cv::dilate(image, dilatedImage, kernel);

    // 进行高斯模糊，平滑笔迹
    cv::Mat blurredImage;
    cv::GaussianBlur(dilatedImage, blurredImage, cv::Size(3, 3), 0);

    // 二值化处理
    cv::Mat binaryImage;
    cv::threshold(blurredImage, binaryImage, 127, 255, cv::THRESH_BINARY);

    // 从处理后的图像中提取修改后的坐标点数组
    std::vector<std::pair<int, int>> outputPoints;
    for (int y = 0; y < binaryImage.rows; ++y) {
        for (int x = 0; x < binaryImage.cols; ++x) {
            if (binaryImage.at<uchar>(y, x) == 255) {
                outputPoints.emplace_back(x, y);
            }
        }
    }

    return outputPoints;
}

int main() {
    // 示例输入坐标点数组
    std::vector<std::pair<int, int>> inputPoints = {
        {10, 10}, {11, 10}, {12, 10}, {13, 10}, {14, 10},
        {10, 11}, {11, 11}, {12, 11}, {13, 11}, {14, 11},
        {10, 12}, {11, 12}, {12, 12}, {13, 12}, {14, 12}
    };

    // 调用函数进行笔迹加宽并平滑处理
    std::vector<std::pair<int, int>> outputPoints = widenAndSmoothStroke(inputPoints);

    // 输出修改后的坐标点数组
    std::cout << "Modified points:" << std::endl;
    for (const auto& point : outputPoints) {
        std::cout << "(" << point.first << ", " << point.second << ")" << std::endl;
    }

    return 0;
}