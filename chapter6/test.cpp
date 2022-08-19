#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../chapter2/metric.h"
#include "../utils/dataset.h"
#include "svm.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;


int main(int argc, char* argv[]) {
    string kernel_name;
    if(argc >= 2) {
        kernel_name = argv[1];
    }
    
    float params[2] = {2.0, 0.0};

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

    SVM svm(2.0, kernel_name, params);
    svm.solve(X_train, y_train, 40);
    // std::cout << "alphas: " << svm.get_alphas().transpose() << std::endl;
    VectorXf pred = svm.predict(X_test);
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
}