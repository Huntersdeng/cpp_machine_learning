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
    string file_path, kernel_name;
    float params[2] = {2.0, 0.0};
    if(argc >= 4) {
        file_path = argv[1];
        kernel_name = argv[2];
        int i = 0;
        while(i + 3 < argc) {
            params[i] = atof(argv[i+3]);
            ++i;
        }
    } else {
        std::cout << "Usage: $(program_name) $(file_path) $(kernel_name)" << std::endl;
        return 0;
    }

    MatrixXf X;
    VectorXf y;
    read_csv2matrix(file_path, X, y);
    int m = X.rows();
    int d = X.cols();
    for(int i=0; i<m; ++i) {
        if(y(i,0) == 0.0) {
            y(i,0) = -1;
        }
    }
    std::cout << "y: " << y.transpose() << std::endl;

    MatrixXf X_data = X.block(0, 0, m, d-1);
    SVM svm(2.0, kernel_name, params);
    svm.solve(X_data, y, 40);
    std::cout << "alphas: " << svm.get_alphas().transpose() << std::endl;
    VectorXf pred = svm.predict(X_data);
    for(int i=0; i<m; ++i) {
        if(pred(i,0) >= 0.0) {
            pred(i,0) = 1.0;
        } else {
            pred(i,0) = -1.0;
        }
    }

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