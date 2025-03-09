#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void displayImg()
{
    //读取一张手写数字图片(28,28)
    Mat image = cv::imread("1.jpg", 0);
    int type = image.type();
    int n = image.rows;
    Mat img_show = image.clone();
    //更换数据类型有uchar->float32
    image.convertTo(image, CV_32F);
    //归一化
    image = image / 255.0;
    //(1,784)
    image = image.reshape(1, 1);
     
    //加载ann模型
    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>("mnist_ann50.xml");
    //预测图片
    Mat pre_out;
    float ret = ann->predict(image, pre_out);
    // 打印当前层的权重和输出
    std::cout << "pre_out----->----------------->----------->----------" << std::endl;
    for (int i = 0; i < pre_out.rows; ++i) {
        for (int j = 0; j < pre_out.cols; ++j) {
            std::cout << pre_out.at<float>(i, j) << " ";
        }
        std::cout << "===================" << std::endl;
    }
    std::cout << "----->----------------->----------->----------" << std::endl;
    double maxVal = 0;
    cv::Point maxPoint;
    cv::minMaxLoc(pre_out, NULL, &maxVal, NULL, &maxPoint);
    int max_index = maxPoint.x;
    cout << "图像上的数字为：" << max_index << " 置信度为：" << maxVal << endl;

    cv::imshow("img", img_show);
    cv::waitKey(0);
    getchar();

}

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //只打印错误信息
    
    /*
    Mat A = Mat::ones(2, 3, CV_32FC1);
    Mat B = Mat::ones(3, 2, CV_32FC1);
    Mat AB;

    A.at<float>(0, 0) = 1;
    A.at<float>(0, 1) = 2;
    A.at<float>(0, 2) = 3;
    A.at<float>(1, 0) = 4;
    A.at<float>(1, 1) = 5;
    A.at<float>(1, 2) = 6;

    B.at<float>(0, 0) = 1;
    B.at<float>(0, 1) = 2;
    B.at<float>(1, 0) = 3;
    B.at<float>(1, 1) = 4;
    B.at<float>(2, 0) = 5;
    B.at<float>(2, 1) = 6;

    AB = A * B;

    cout << "A=\n" << A << endl << endl;
    cout << "B=\n" << B << endl << endl;
    cout << "AB=\n" << AB << endl << endl;
    */

    displayImg();//
   return 0; //
   
}