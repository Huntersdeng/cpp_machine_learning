#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../utils/dataset.h"
#include "../chapter2/metric.h"
#include "dtree.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;
using Eigen::VectorXi;

int main(int argc, char* argv[]) {
    string file_path = "../watermelon.csv";
    int method = GINI_INDEX;
    if(argc > 2) {
        file_path = argv[1];
        method = atoi(argv[2]);
    }
    vector<vector<float>> X_data;
    vector<int> y_data;
    read_csv2vector(file_path, X_data, y_data);
    for(vector<float> &x: X_data) {
        x.pop_back();
        x.pop_back();
    }
    int m = X_data.size();
    int d = X_data[0].size();

    DTree tree(method);
    tree.solve(X_data, y_data);
    vector<int> prediction = tree.predict(X_data);
    VectorXf pred(m, 1), y(m, 1);
    for(int i=0; i<m; ++i) {
        pred(i, 0) = prediction[i];
        y(i, 0) = y_data[i];
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