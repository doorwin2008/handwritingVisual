#include<iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;


cv::Mat sigmoid(const cv::Mat& input) {
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
                output.at<float>(i, j) = 1.0 / (1.0 + std::exp(0.1 * -val  ));
            }
        }
    }
    return output;
}

void displayImg()
{
    //读取一张手写数字图片(28,28)
    Mat image = cv::imread("5.jpg", 0);
    Mat img_show = image.clone();
    //更换数据类型有uchar->float32
    image.convertTo(image, CV_32F);
    //归一化
    image = image / 255.0;
    //(1,784)
    image = image.reshape(1, 1);

    //加载ann模型
    //cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>("mnist_ann -20.xml");
    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>("mnist_ann20.xml");

    // 获取模型的层数
    //权重的计算，从第二层开始，bias偏移量保存在最后一行
    //
    cv::Mat layerSizes = ann->getLayerSizes();
    vector<int> layer_sizes;
    for (int i = 0; i < layerSizes.rows; ++i) {
        for (int j = 0; j < layerSizes.cols; ++j) {
            //std::cout << layerSizes.at<int>(i, j) << " ";
            layer_sizes.push_back(layerSizes.at<int>(i, j));
        }
        //std::cout << "===================" << std::endl;
    }
    //std::cout << "-------------------------------------------" << std::endl;

    int layerCount = layerSizes.rows ;

    // 手动前向传播
    vector<Mat> layerOutputs;
    layerOutputs.push_back(image); // 输入层,把识别的图片矩阵导入一个矩阵向量

    // 获取每一层的权重矩阵
    vector<Mat> weights;
    Mat weightMat = ann->getWeights(0);
    //***** 第一层权重特殊处理。
    weights.push_back(weightMat);

    for (int i = 1; i < layerCount; ++i) {
        Mat weightMat = ann->getWeights(i);
        Mat w = weightMat.colRange(0, layer_sizes[i]);//***** 这里根据权重文件中过去的layer size读取实际的权重大小。
        weights.push_back(w);
    }
   
    
    //cv::Mat weight748 =cv::Mat::ones(1, 784, CV_32F);;
    //weights[0].reshape(1, 2).copyTo(matrix_2x784);
    //std::cout << "matrix_2x784 " << std::endl;
    for (int jj = 0; jj < layerOutputs[0].cols; jj++) {
        double tt = layerOutputs[0].at<float>(0, jj);
        layerOutputs[0].at<float>(0, jj) = weights[0].at<double>(0, (2 * jj + 1)) + tt* weights[0].at<double>(0, (2 * jj));
    }

    // 手动计算每一层的输出
    for (int i = 1; i < layerCount ; ++i) {
        Mat input = layerOutputs.back();//返回容器中最后一个元素
        Mat weight = weights[i];
        if (weight.type() != CV_32F) {
            weight.convertTo(weight, CV_32F);
        }

        Mat output;
 
       // 提取偏置
       Mat bias = weight.row(weight.rows-1);//最后一行作为偏置
       std::cout << "base of :" << i << std::endl;
       for (int j = 0; j < bias.cols; ++j) {
           std::cout << bias.at<float>(0, j) << " ";
       }
       std::cout << "-------------------------------------------" << std::endl;
       // 提取真正的权重
       Mat realWeight = weight.rowRange(0, weight.rows-1);
       Mat temp = realWeight;
       output = input * temp ;//+ bias

        // 应用激活函数（这里假设使用ReLU）
        //cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);

        layerOutputs.push_back(output);

        // 打印当前层的权重和输出
        std::cout << "weight of "<< i << std::endl;
        for (int ii = 0; ii < output.rows; ++ii) {
            for (int jj = 0; jj < output.cols; ++jj) {
                std::cout << output.at<float>(ii, jj) << " ";
            }
            std::cout << "===================" << std::endl;
        }
        std::cout << "----->----------------->----------->----------" << std::endl;
    }

    // 最后一层的输出即为预测结果
    Mat pre_out = layerOutputs.back();

    //Mat t = pre_out.t();
    //Mat output_one_row = t.reshape(1, 1);
    cv::Mat normalized_data = sigmoid(pre_out);//归一化处理
    std::cout << "normalized_data-----> "<< normalized_data.type() << std::endl;
    for (int i = 0; i < normalized_data.rows; ++i) {
        for (int j = 0; j < normalized_data.cols; ++j) {
            std::cout << normalized_data.at<float>(i, j) << " ";
        }
        std::cout << "===================" << std::endl;
    }
    std::cout << "----->----------------->----------->----------" << std::endl;/**/
    double maxVal = 0;
    cv::Point maxPoint;
    cv::minMaxLoc(normalized_data, NULL, &maxVal, NULL, &maxPoint);

    int max_index = maxPoint.x;

    std::cout << "图像上的数字为：" << max_index << " 置信度为：" << maxVal << endl;

    cv::imshow("img", img_show);
    cv::waitKey(0);
    getchar();
}

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //只打印错误信息

    displayImg();

    return 0;
}