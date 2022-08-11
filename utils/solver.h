#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;


class Newton {
public:
    inline static void step(VectorXf &w, VectorXf &first_derv, MatrixXf &second_derv) {
        w = w - second_derv.inverse() * first_derv;
    }
};

class GradDesc {
public:
    inline static void step(VectorXf &w, VectorXf &first_derv, float lr=0.001) {
        w = w - lr * first_derv;
    }
};