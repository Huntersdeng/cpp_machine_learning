#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../utils/dataset.h"
#include "../chapter2/metric.h"
#include "kmeans.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::Matrix2i;
using Eigen::VectorXf;
using Eigen::VectorXi;

int main(int argc, char* argv[]) {
    string file_path = "../dataset/watermelon.csv";
    if(argc > 1) {
        file_path = argv[1];
    }
    vector<vector<float>> X_data;
    vector<int> y_data;
    read_csv2vector(file_path, X_data, y_data);
    int m = X_data.size();
    int d = X_data[0].size();

    MatrixXf X(m, 2);
    VectorXf pred(m, 1), y(m, 1);

    for(int i=0; i<m; ++i) {
        X(i, 0) = X_data[i][6];
        X(i, 1) = X_data[i][7];
        y(i, 0) = y_data[i];
    } 

    KMeans km(2);
    km.solve(X, pred);

    /* Result */
    Matrix2i mat;
    Metric::confusion_matrix(pred, y, mat);
    std::cout << "labels:      " << y.transpose() << std::endl;
    std::cout << "Predictions: " << pred.transpose() << std::endl;
    std::cout << "JC: " << Metric::jaccard_coefficient(pred, y) << std::endl;
    std::cout << "FMI: " << Metric::fmi(pred, y) << std::endl;
    
}