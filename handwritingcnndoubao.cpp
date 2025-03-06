#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <opencv2/dnn.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;

// 小端存储转换
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// 假设已经有一个训练好的 Net 对象 net
void saveModelAsONNX(const Net& net, const string& filename) {
    // 定义输入层的名称和形状
    vector<String> inputBlobNames = { "input" };
    vector<MatShape> inputShapes = { MatShape{1, 1, 28, 28} }; // 示例输入形状，根据实际情况调整

    // 导出为 ONNX 格式
    //net.writeToONNX(filename, inputBlobNames, inputShapes);
    cout << "Model saved as ONNX: " << filename << endl;
}
// 假设已经有一个训练好的 Net 对象 net
void saveModelAsCaffe(Net& net, const string& prototxtFileName, const string& caffemodelFileName) {
    // 保存模型结构到 .prototxt 文件
    ofstream prototxtFile(prototxtFileName);
    if (!prototxtFile.is_open()) {
        cerr << "Failed to open file for writing prototxt: " << prototxtFileName << endl;
        return;
    }
    prototxtFile << net.dump();
    prototxtFile.close();

    // 保存模型权重到 .caffemodel 文件
    //net.save(caffemodelFileName);
    //cout << "Model saved as Caffe: " << prototxtFileName << " and " << caffemodelFileName << endl;
}


// 读取image数据集信息
Mat read_mnist_image(const string fileName) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    Mat DataMat;

    ifstream file(fileName, ios::binary);
    if (file.is_open()) {
        cout << "成功打开图像集 ..." << endl;

        file.read((char*)&magic_number, sizeof(magic_number)); // 幻数（文件格式）
        if (!file.good()) {
            cerr << "Error reading magic number from " << fileName << endl;
            return Mat();
        }
        file.read((char*)&number_of_images, sizeof(number_of_images)); // 图像总数
        if (!file.good()) {
            cerr << "Error reading number of images from " << fileName << endl;
            return Mat();
        }
        file.read((char*)&n_rows, sizeof(n_rows)); // 每个图像的行数
        if (!file.good()) {
            cerr << "Error reading number of rows from " << fileName << endl;
            return Mat();
        }
        file.read((char*)&n_cols, sizeof(n_cols)); // 每个图像的列数
        if (!file.good()) {
            cerr << "Error reading number of columns from " << fileName << endl;
            return Mat();
        }

        magic_number = reverseInt(magic_number);
        number_of_images = reverseInt(number_of_images);
        n_rows = reverseInt(n_rows);
        n_cols = reverseInt(n_cols);
        cout << "幻数（文件格式）:" << magic_number
            << " 图像总数:" << number_of_images
            << " 每个图像的行数:" << n_rows
            << " 每个图像的列数:" << n_cols << endl;

        cout << "开始读取Image数据......" << endl;

        DataMat = Mat::zeros(number_of_images, n_rows * n_cols, CV_32F);//CV_32FC1
        for (int i = 0; i < number_of_images; i++) {
            for (int j = 0; j < n_rows * n_cols; j++) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                if (!file.good()) {
                    cerr << "Error reading pixel data from " << fileName << " at row " << i << ", col " << j << endl;
                    return Mat();
                }
                float pixel_value = float(temp);
                DataMat.at<float>(i, j) = pixel_value / 255.0;
            }
        }
        cout << "读取Image数据完毕......" << endl;
    }
    file.close();
    return DataMat;
}

// 读取label数据集信息
Mat read_mnist_label(const string fileName) {
    int magic_number;
    int number_of_items;

    Mat LabelMat;

    ifstream file(fileName, ios::binary);
    if (file.is_open()) {
        cout << "成功打开标签集 ... " << endl;

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

        cout << "幻数（文件格式）:" << magic_number << "  ;标签总数:" << number_of_items << endl;

        cout << "开始读取Label数据......" << endl;
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
        cout << "读取Label数据完毕......" << endl;
    }
    file.close();
    return LabelMat;
}

// 将标签数据改为one-hot型
Mat one_hot(Mat label, int classes_num) {
    int rows = label.rows;
    Mat one_hot = Mat::zeros(rows, classes_num, CV_32FC1);
    for (int i = 0; i < label.rows; i++) {
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

// 创建CNN网络
void create_cnn_network(Net& net) {
    LayerParams lp;

    //1 第一个卷积层
    lp.name = "conv1";
    lp.type = "Convolution";
    lp.blobs.resize(2);
    lp.set("num_output", 16);
    lp.set("kernel_size", 5);
    lp.set("stride", 1);
    lp.set("pad", 2);
    net.addLayer(lp.name, lp.type, lp);

    //2 ReLU 激活函数
    lp.name = "relu1";
    lp.type = "ReLU";
    net.addLayer(lp.name, lp.type, lp);

    //3 最大池化层
    lp.name = "pool1";
    lp.type = "Pooling";
    lp.set("kernel_size", 2);
    lp.set("stride", 2);
    lp.set("pool", "MAX");
    net.addLayer(lp.name, lp.type, lp);

    //4 第二个卷积层
    lp.name = "conv2";
    lp.type = "Convolution";
    lp.blobs.resize(2);
    lp.set("num_output", 32);
    lp.set("kernel_size", 5);
    lp.set("stride", 1);
    lp.set("pad", 2);
    net.addLayer(lp.name, lp.type, lp);

    //5 ReLU 激活函数
    lp.name = "relu2";
    lp.type = "ReLU";
    net.addLayer(lp.name, lp.type, lp);

    //6 最大池化层
    lp.name = "pool2";
    lp.type = "Pooling";
    lp.set("kernel_size", 2);
    lp.set("stride", 2);
    lp.set("pool", "MAX");
    net.addLayer(lp.name, lp.type, lp);

    //7 展平层
    lp.name = "flatten";
    lp.type = "Flatten";
    net.addLayer(lp.name, lp.type, lp);
/*
    //8 全连接层
    lp.name = "fc1";
    lp.type = "InnerProduct";
    lp.blobs.resize(2);
    lp.set("num_output", 128);
    net.addLayer(lp.name, lp.type, lp);

    //9 ReLU 激活函数
    lp.name = "relu3";
    lp.type = "ReLU";
    net.addLayer(lp.name, lp.type, lp);

    //10 输出层
    lp.name = "fc2";
    lp.type = "InnerProduct";
    lp.blobs.resize(2);
    lp.set("num_output", 10);
    net.addLayer(lp.name, lp.type, lp);

    //11 Softmax 激活函数
    lp.name = "softmax";
    lp.type = "Softmax";
    net.addLayer(lp.name, lp.type, lp);*/
}

// 手动实现反向传播和参数更新
void backpropagate(Net& net, vector<Mat>& layerOutputs, Mat& output, Mat& labels, double learningRate) {
    // 计算输出层的梯度（交叉熵损失的梯度）
    Mat dOutput = output - labels;

    // 获取所有层的名称
    vector<String> layerNames = net.getLayerNames();
    // 从最后一层开始反向传播
    for (int i = static_cast<int>(layerNames.size()) - 1; i > 0; --i) {
        Ptr<Layer> layer = net.getLayer(net.getLayerId(layerNames[i]));
        Mat input = layerOutputs[i - 1];

        if (layer->type == "InnerProduct" || layer->type == "Convolution") {
            // 计算权重和偏置的梯度
            Mat dWeights, dBias;

            if (layer->type == "InnerProduct") {
                // 全连接层的权重和偏置梯度计算
                gemm(input.t(), dOutput, 1.0, Mat(), 0.0, dWeights);
                reduce(dOutput, dBias, 0, REDUCE_SUM);
            }
            else if (layer->type == "Convolution") {
                // 卷积层的权重和偏置梯度计算，这里简化处理，实际需要更复杂的实现
                // 简单示例：使用输入和输出梯度计算权重梯度
                dWeights = Mat::zeros(layer->blobs[0].size(), CV_32F);
                dBias = Mat::zeros(layer->blobs[1].size(), CV_32F);
                // 这里省略具体的卷积层梯度计算细节
            }

            // 更新权重和偏置
            layer->blobs[0] -= learningRate * dWeights;
            layer->blobs[1] -= learningRate * dBias;

            // 计算前一层的梯度
            Mat dInput;
            gemm(dOutput, layer->blobs[0].t(), 1.0, Mat(), 0.0, dInput);
            dOutput = dInput;
        }
        else if (layer->type == "ReLU") {
            // ReLU 激活函数的梯度计算
            Mat mask = input > 0;
            dOutput = dOutput.mul(mask);
        }
    }
}

int main() {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR); // 只打印错误信息
    system("chcp 65001");
    /*
    ---------第一部分：训练数据准备-----------
    */
    // 读取训练标签数据 (60000,1) 类型为int32
    Mat train_labels = read_mnist_label("D:/doorw_source/CNNHandwriting-main/data/train-labels.idx1-ubyte");
    if (train_labels.empty()) {
        cerr << "Failed to read training labels." << endl;
        return -1;
    }
    // ann神经网络的标签数据需要转为one-hot型
    train_labels = one_hot(train_labels, 10);

    // 读取训练图像数据 (60000,784) 类型为float32 数据未归一化
    Mat train_images = read_mnist_image("D:/doorw_source/CNNHandwriting-main/data/train-images.idx3-ubyte");
    if (train_images.empty()) {
        cerr << "Failed to read training images." << endl;
        return -1;
    }

    // 调整训练图像形状以便输入到网络中 (60000, 1, 28, 28)
    train_images = train_images.reshape(4, train_images.rows);

    // 读取测试数据标签(10000,1) 类型为int32 测试标签不用转为one-hot型
    Mat test_labels = read_mnist_label("D:/doorw_source/CNNHandwriting-main/data/t10k-labels.idx1-ubyte");
    if (test_labels.empty()) {
        cerr << "Failed to read test labels." << endl;
        return -1;
    }

    // 读取测试数据图像 (10000,784) 类型为float32 数据未归一化
    Mat test_images = read_mnist_image("D:/doorw_source/CNNHandwriting-main/data/t10k-images.idx3-ubyte");
    if (test_images.empty()) {
        cerr << "Failed to read test images." << endl;
        return -1;
    }

    // 调整测试图像形状以便输入到网络中 (10000, 1, 28, 28)
    test_images = test_images.reshape(4, test_images.rows);

    /*
    ---------第二部分：构建CNN训练模型并进行训练-----------
    */
    Net net;
    create_cnn_network(net);

    // 设置训练参数
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 初始化训练参数
    double learningRate = 0.001;
    int maxIter = 10;
    int batchSize = 100;

    // 开始训练
    cout << "开始进行训练..." << endl;
    for (int iter = 0; iter < maxIter; ++iter) {
        for (int i = 0; i < train_images.rows; i += batchSize) {
            int end = min(i + batchSize, train_images.rows);
            Mat batchImages = train_images.rowRange(i, end);
            Mat batchLabels = train_labels.rowRange(i, end);

            cout << "Batch images shape before input: " << batchImages.size() << endl;
            net.setInput(batchImages);
            cout << "Input batch images shape: " << batchImages.size() << endl;
            vector<Mat> layerOutputs;
            layerOutputs.push_back(batchImages);

            // 正向传播并记录每一层的输出
            vector<String> layerNames = net.getLayerNames();
            for (const auto& layerName : layerNames) {
                Mat output = net.forward(layerName);
                layerOutputs.push_back(output);
                int id = net.getLayerId(layerName);
                auto layer = net.getLayer(id);
                printf("layer id:%d, type: %s, name:%s \n", id, layer->type.c_str(), layer->name.c_str());
                
            }
            Mat output = layerOutputs.back();

            // 计算损失（交叉熵损失）
            Mat loss = Mat::zeros(1, 1, CV_32F);
            for (int j = 0; j < batchLabels.rows; ++j) {
                for (int k = 0; k < batchLabels.cols; ++k) {
                    float label = batchLabels.at<float>(j, k);
                    float pred = output.at<float>(j, k);
                    loss.at<float>(0, 0) += -label * log(pred);
                }
            }
            loss /= batchSize;

            // 反向传播和参数更新
            backpropagate(net, layerOutputs, output, batchLabels, learningRate);

            if ((i / batchSize) % 100 == 0) {
                cout << "Iteration " << iter << ", Batch " << i / batchSize << ", Loss: " << loss.at<float>(0, 0) << endl;
            }
        }
    }
    cout << "训练完成" << endl;
    //saveModelAsONNX(net, "model.onnx");
    saveModelAsCaffe(net, "model.prototxt", "model.caffemodel");
    /*
    ---------第三部分：在测试数据集上预测计算准确率-----------
    */
    Mat pre_out;
    // 返回值为第一个图像的预测值 pre_out为整个batch的预测值集合
    cout << "开始进行预测..." << endl;
    for (int i = 0; i < test_images.rows; i += batchSize) {
        int end = min(i + batchSize, test_images.rows);
        Mat batch = test_images.rowRange(i, end);
        net.setInput(batch);
        Mat out = net.forward();
        pre_out.push_back(out);
    }
    cout << "预测完成" << endl;

    // 计算准确率
    int equal_nums = 0;
    for (int i = 0; i < pre_out.rows; i++) {
        Point maxLoc;
        minMaxLoc(pre_out.row(i), nullptr, nullptr, nullptr, &maxLoc);
        int predictedLabel = maxLoc.x;
        if (predictedLabel == test_labels.at<int32_t>(i, 0)) {
            equal_nums++;
        }
    }
    double accuracy = (double)equal_nums / test_labels.rows;
    cout << "准确率: " << accuracy * 100 << "%" << endl;

    return 0;
}