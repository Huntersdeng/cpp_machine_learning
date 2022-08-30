#include "kernel.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

using Eigen::VectorXf;
using Eigen::MatrixXf;

using std::vector;

VectorXf LinearKernel::operator() (const MatrixXf &x1, const VectorXf& x2) {
    return x1 * x2;
}

VectorXf PolyKernel::operator() (const MatrixXf &x1, const VectorXf& x2) {
    return (x1 * x2).array().pow(d);
}

VectorXf RBFKernel::operator() (const MatrixXf &x1, const VectorXf& x2) {
    int m = x1.rows();
    VectorXf res(m);
    for(int i=0; i<m; ++i) {
        VectorXf x = x1.row(i);
        res(i) = (*this)(x, x2);
    }
    return res;
}

VectorXf LaplaceKernel::operator() (const MatrixXf &x1, const VectorXf& x2) {
    int m = x1.rows();
    VectorXf res(m);
    for(int i=0; i<m; ++i) {
        VectorXf x = x1.row(i);
        res(i) = (*this)(x, x2);
    }
    return res;
}

VectorXf SigmoidKernel::operator() (const MatrixXf &x1, const VectorXf& x2) {
    int m = x1.rows();
    VectorXf res(m);
    for(int i=0; i<m; ++i) {
        VectorXf x = x1.row(i);
        res(i) = (*this)(x, x2);
    }
    return res;
}

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

shared_ptr<BaseKernel> LinearKernelFactory::create() {
    return std::make_shared<LinearKernel>();
}

shared_ptr<BaseKernel> PolyKernelFactory::create() {
    int d = (int)param[0];
    return std::make_shared<PolyKernel>(d);
}

shared_ptr<BaseKernel> RBFKernelFactory::create() {
    float sigma = param[0];
    return std::make_shared<RBFKernel>(sigma);
}

shared_ptr<BaseKernel> LaplaceKernelFactory::create() {
    float sigma = param[0];
    return std::make_shared<LaplaceKernel>(sigma);
}

shared_ptr<BaseKernel> SigmoidKernelFactory::create() {
    float beta = param[0], theta = param[1];
    return std::make_shared<SigmoidKernel>(beta, theta);
}