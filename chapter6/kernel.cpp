#include "kernel.h"

#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::VectorXf;
using Eigen::MatrixXf;

float RBFKernel::operator() (const VectorXf &x1, const VectorXf& x2) {
    float squared_norm = (x1 - x2).squaredNorm();
    return exp(-squared_norm / (2 * sigma * sigma));
}

float LaplaceKernel::operator() (const VectorXf &x1, const VectorXf& x2) {
    float norm = (x1 - x2).norm();
    return exp(-norm / sigma);
}

float SigmoidKernel::operator() (const VectorXf &x1, const VectorXf& x2) {
    float linear = x1.transpose() * x2;
    return tanh(beta * linear + theta);
}