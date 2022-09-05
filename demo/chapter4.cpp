#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../utils/dataset.h"
#include "../chapter2/metric.h"
#include "../chapter4/dtree.h"
#include "../ml.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;
using Eigen::VectorXi;

int main(int argc, char* argv[]) {
    principle method = principle::ENT;
    if (argc > 1) {
        if(atoi(argv[1]) == 1) {
            method = principle::GINI;
        } else {
            method = principle::ENT;
        }
    }

    MatrixXf X_train, X_test;
    VectorXf y_train, y_test;

    /* Using watermelon dataset */
    read_csv2matrix("./dataset/watermelon.csv", X_train, y_train);

    X_train = X_train.block(0, 0, 17, 6);

    X_test = X_train; y_test = y_train;

    Classifier* tree = new DTree(method);
    tree->fit(X_train, y_train);
    VectorXf pred = tree->predict(X_test);

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
    delete tree;
}