#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../chapter2/metric.h"
#include "../utils/dataset.h"
#include "../chapter6/svm.h"
#include "../ml.h"
#include "../chapter6/kernel.h"

using std::vector;
using std::string;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;


int main(int argc, char* argv[]) {
    string kernel_name;
    std::cout << "Kernel (option: linear, poly, rbf, laplace, sigmoid) :\n";
    std::cin >> kernel_name;
    shared_ptr<KernelFactory> factory;
    vector<float> param;
    float p;
    if (kernel_name == "linear") {
        std::cout << "Using linear kernel\n";
        factory = std::make_shared<LinearKernelFactory>(param);
    } else if(kernel_name == "poly") {
        std::cout << "Using poly kernel\n";
        std::cout << "d >= 1:";
        std::cin >> p;
        param.push_back(p);
        factory = std::make_shared<PolyKernelFactory>(param);
    } else if(kernel_name == "rbf") {
        std::cout << "Using rbf kernel\n";
        std::cout << "sigma > 0:";
        std::cin >> p;
        param.push_back(p);
        factory = std::make_shared<RBFKernelFactory>(param);
    } else if(kernel_name == "laplace") {
        std::cout << "Using laplace kernel\n";
        std::cout << "sigma > 0:";
        std::cin >> p;
        param.push_back(p);
        factory = std::make_shared<LaplaceKernelFactory>(param);
    } else if(kernel_name == "sigmoid") {
        std::cout << "Using sigmoid kernel\n";
        std::cout << "beta > 0:";
        std::cin >> p;
        param.push_back(p);
        std::cout << "theta < 0:";
        std::cin >> p;
        param.push_back(p);
        factory = std::make_shared<SigmoidKernelFactory>(param);
    } else {
        std::cout << "Using linear kernel\n";
        factory = std::make_shared<LinearKernelFactory>(param);
    }


    MatrixXf X_train, X_test;
    VectorXf y_train, y_test;
    read_csv2matrix("./dataset/watermelon.csv", X_train, y_train);
    X_test = X_train;
    y_test = y_train;
    // vector<int> idx_train, idx_test;
    // read_Mnist_Label("../dataset/mnist/train-labels.idx1-ubyte", y_train, 0, 1, idx_train);
    // read_Mnist_Label("../dataset/mnist/t10k-labels.idx1-ubyte", y_test, 0, 1, idx_test);
    // read_Mnist_Images("../dataset/mnist/train-images.idx3-ubyte", X_train, idx_train);
    // read_Mnist_Images("../dataset/mnist/t10k-images.idx3-ubyte", X_test, idx_test);
    int m = X_train.rows();
    int m_test = X_test.rows();
    int d = X_train.cols();
    int d_test = X_test.cols();
    std::cout << m << " " << d << " " << m_test << " " << d_test << std::endl;
    for(int i=0; i<m; ++i) {
        if(y_train(i,0) == 0.0) {
            y_train(i,0) = -1;
        }
    }
    for(int i=0; i<m_test; ++i) {
        if(y_test(i,0) == 0.0) {
            y_test(i,0) = -1;
        }
    }
    Classifier* model = new SVM(2.0, 40, factory);
    model->fit(X_train, y_train);
    // std::cout << "alphas: " << svm.get_alphas().transpose() << std::endl;
    VectorXf pred = model->predict(X_test);
    for(int i=0; i<m_test; ++i) {
        if(pred(i,0) >= 0.0) {
            pred(i,0) = 1.0;
        } else {
            pred(i,0) = -1.0;
        }
    }

    /* Result */
    Matrix2i mat;
    Metric::confusion_matrix(pred, y_test, mat);
    std::cout << "labels:      " << y_test.transpose() << std::endl;
    std::cout << "Predictions: " << pred.transpose() << std::endl;
    std::cout << "MSE: " << Metric::mse(pred, y_test) << std::endl;
    std::cout << "Accuracy: " << Metric::accuracy(pred, y_test) << std::endl;
    std::cout << "Error_rate: " << Metric::error_rate(pred, y_test) << std::endl;
    std::cout << "Confusion matrix:\n" << mat << std::endl;
    std::cout << "Precision: " << Metric::precision(pred, y_test) << std::endl;
    std::cout << "Recall: " << Metric::recall(pred, y_test) << std::endl;
    std::cout << "F1_score: " << Metric::f1_score(pred, y_test) << std::endl;
    delete model;
}