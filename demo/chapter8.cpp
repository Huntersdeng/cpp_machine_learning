#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../ml.h"
#include "../utils/dataset.h"
#include "../chapter2/metric.h"
#include "../chapter8/kmeans.h"
#include "../chapter8/lvq.h"
#include "../chapter8/gauss_mixture.h"

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
    MatrixXf X_train, X_test;
    VectorXf y_train, y_test;

    /* Using watermelon dataset */
    read_csv2matrix("./dataset/watermelon.csv", X_train, y_train);

    X_train = X_train.block(0, 6, 17, 2);

    X_test = X_train; y_test = y_train;

    Cluster* model = new GaussianMixture(2, 5);
    model->fit(X_train);
    VectorXf pred = model->predict(X_test);

    /* Result */
    Matrix2i mat;
    Metric::confusion_matrix(pred, y_test, mat);
    std::cout << "labels:      " << y_test.transpose() << std::endl;
    std::cout << "Predictions: " << pred.transpose() << std::endl;
    std::cout << "JC: " << Metric::jaccard_coefficient(pred, y_test) << std::endl;
    std::cout << "FMI: " << Metric::fmi(pred, y_test) << std::endl;
    delete model;
}