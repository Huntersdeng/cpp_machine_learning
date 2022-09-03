#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../ml.h"
#include "../chapter2/metric.h"
#include "../chapter3/linear_reg.h"
#include "../chapter3/logistic_reg.h"
#include "../chapter3/LDA.h"
#include "../utils/dataset.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;


int main(int argc, char* argv[]) {
    int label0 = 0, label1 = 1;
    if(argc >= 3) {
        label0 = atoi(argv[1]);
        label1 = atoi(argv[2]);
    }
    MatrixXf X_train, X_test;
    VectorXf y_train, y_test;

    /* Using watermelon dataset */
    read_csv2matrix("./dataset/watermelon.csv", X_train, y_train);
    X_test = X_train; y_test = y_train;

    /* Using mnist dataset */
    // vector<int> idx_train, idx_test;
    // read_Mnist_Label("./dataset/mnist/train-labels.idx1-ubyte", y_train, label0, label1, idx_train);
    // read_Mnist_Label("./dataset/mnist/t10k-labels.idx1-ubyte", y_test, label0, label1, idx_test);
    // read_Mnist_Images("./dataset/mnist/train-images.idx3-ubyte", X_train, idx_train);
    // read_Mnist_Images("./dataset/mnist/t10k-images.idx3-ubyte", X_test, idx_test);
    
    int m = X_train.rows();
    int m_test = X_test.rows();
    int d = X_train.cols();
    int d_test = X_test.cols();
    std::cout << m << " " << d << " " << m_test << " " << d_test << std::endl;

    /* logistric regression */
    Classifier* model = new LinearReg();
    model->fit(X_train, y_train);
    // std::cout << lr.get_w().transpose() << std::endl;
    VectorXf pred = model->predict(X_test);

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