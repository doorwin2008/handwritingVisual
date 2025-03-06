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


// �������ڽ��ʼ��ӿ�ƽ��
std::vector<std::pair<int, int>> widenAndSmoothStroke(const std::vector<std::pair<int, int>>& inputPoints) {
    // ����һ�� 28x28 �Ŀհ�ͼ��
    cv::Mat image = cv::Mat::zeros(28, 28, CV_8UC1);

    // ��������������Ƶ�ͼ����
    for (const auto& point : inputPoints) {
        int x = point.first;
        int y = point.second;
        if (x >= 0 && x < 28 && y >= 0 && y < 28) {
            image.at<uchar>(y, x) = 255;
        }
    }

    // �������ͺ�
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // ������̬ѧ���Ͳ������ӿ�ʼ�
    cv::Mat dilatedImage;
    cv::dilate(image, dilatedImage, kernel);

    // ���и�˹ģ����ƽ���ʼ�
    cv::Mat blurredImage;
    cv::GaussianBlur(dilatedImage, blurredImage, cv::Size(3, 3), 0);

    // ��ֵ������
    cv::Mat binaryImage;
    cv::threshold(blurredImage, binaryImage, 127, 255, cv::THRESH_BINARY);

    // �Ӵ�����ͼ������ȡ�޸ĺ�����������
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
    // ʾ���������������
    std::vector<std::pair<int, int>> inputPoints = {
        {10, 10}, {11, 10}, {12, 10}, {13, 10}, {14, 10},
        {10, 11}, {11, 11}, {12, 11}, {13, 11}, {14, 11},
        {10, 12}, {11, 12}, {12, 12}, {13, 12}, {14, 12}
    };

    // ���ú������бʼ��ӿ�ƽ������
    std::vector<std::pair<int, int>> outputPoints = widenAndSmoothStroke(inputPoints);

    // ����޸ĺ�����������
    std::cout << "Modified points:" << std::endl;
    for (const auto& point : outputPoints) {
        std::cout << "(" << point.first << ", " << point.second << ")" << std::endl;
    }

    return 0;
}