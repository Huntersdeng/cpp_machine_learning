#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../chapter2/metric.h"
#include "linear_reg.h"
#include "logistic_reg.h"
#include "LDA.h"
#include "../utils/dataset.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;


int main() {
    MatrixXf X;
    VectorXf y;
    read_csv2matrix("./watermelon.csv", X, y);
    int m = X.rows();
    int d = X.cols();

    /* logistric regression */
    // LogisticReg lr(d);
    // lr.solve(X, y, 20);
    // std::cout << lr.get_w().transpose() << std::endl;
    // VectorXf pred = lr.predict(X);

    /* LDA */
    MatrixXf X_data = X.block(0, 0, m, d-1);
    LDA lda(d-1);
    lda.solve(X_data, y);
    VectorXf pred = lda.predict(X_data);

    /* Result */
    Matrix2i mat;
    Metric::confusion_matrix(pred, y, mat);
    std::cout << "labels:      " << y.transpose() << std::endl;
    std::cout << "Predictions: " << pred.transpose() << std::endl;
    std::cout << "MSE: " << Metric::mse(pred, y) << std::endl;
    std::cout << "Accuracy: " << Metric::accuracy(pred, y) << std::endl;
    std::cout << "Error_rate: " << Metric::error_rate(pred, y) << std::endl;
    std::cout << "Confusion matrix:\n" << mat << std::endl;
    std::cout << "Precision: " << Metric::precision(pred, y) << std::endl;
    std::cout << "Recall: " << Metric::recall(pred, y) << std::endl;
    std::cout << "F1_score: " << Metric::f1_score(pred, y) << std::endl;
}