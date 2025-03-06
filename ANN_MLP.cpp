#include "opencv2/opencv.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/core/utils/logger.hpp>
using namespace cv;
using namespace ml;
using namespace std;

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); //只打印错误信息
    //训练数据及对应标签
    float trainingData[8][2] = { {480,500},{50,130},{110,32},{490,60},{60,190},{200,189},{78,256},{45,315} };
    float labels[8] = { 1,0,0,1,0,0,0,0 };
    Mat trainingDataMat(8, 2, CV_32FC1, trainingData);
    Mat labelsMat(8, 1, CV_32FC1, labels);
    //建立模型
    Ptr<ANN_MLP> model = ANN_MLP::create();
    //共5层：输入层+3个隐藏层+1个输出层，输入层、隐藏层每层均为2个perceptron  
    Mat layerSizes = (Mat_<int>(1, 5) << 2, 2, 2, 2, 1);
    //设置各层的神经元个数
    model->setLayerSizes(layerSizes);
    //激活函数
    model->setActivationFunction(ANN_MLP::SIGMOID_SYM);
    //MLP的训练方法
    model->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.9);
    //训练模型
    Ptr<TrainData> trainData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
    model->train(trainData);

    Mat src = Mat::zeros(512, 512, CV_8UC3);
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1, 2) << i, j);
            Mat responseMat;
            model->predict(sampleMat, responseMat);
            float p = responseMat.ptr<float>(0)[0];
            if (p > 0.5)
                src.at<Vec3b>(j, i) = Vec3b(0, 125, 125);
            else
                src.at<Vec3b>(j, i) = Vec3b(125, 125, 0);
        }
    }
    //绘制出训练数据在图中的分布
    Mat dst1 = src.clone();
    for (int i = 0; i < sizeof(trainingData[0]); i++)
    {
        float q = labels[i];
        //根据训练数据对应的标签不同绘制不同的颜色：1对应红色，0对应绿色
        if (q == 1)
            circle(dst1, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(0, 0, 255), -1, 8);
        else
            circle(dst1, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(0, 255, 0), -1, 8);
    }
    //在原图像范围内随机生成20个点，并使用训练好的神经网络对其进行预测，并绘制出预测结果
    Mat dst2 = src.clone();
    for (int i = 0; i < 50; i++)
    {

        RNG rngx(i);
        float x = rngx.uniform(0, 512);
        RNG rngy(i * i);
        float y = rngy.uniform(0, 512);

        Mat trainingDataMat = (Mat_<float>(1, 2) << x, y);
        Mat response;
        model->predict(trainingDataMat, response);
        float q = response.ptr<float>(0)[0];
        if (q > 0.5)
            circle(dst2, Point(x, y), 5, Scalar(0, 0, 255), -1, 8);
        else
            circle(dst2, Point(x, y), 5, Scalar(0, 255, 0), -1, 8);
    }

    imshow("output1", dst1);
    imshow("output2", dst2);

    waitKey(0);

    return 0;
}