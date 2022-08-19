#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../utils/solver.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;

void sigmoid(VectorXf&, VectorXf&);

class LogisticReg {
public:
    LogisticReg(int d){
        init(d);
    }
    VectorXf predict(MatrixXf&);
    void solve(MatrixXf&, VectorXf&, int, float);
    inline VectorXf get_w() { return w; }
private:
    void first_derivative(MatrixXf&, VectorXf&, VectorXf&);
    void second_derivative(MatrixXf&, VectorXf&, MatrixXf&);
    void init(int, float=0.0, float=10.0);
    VectorXf w;
};

void LogisticReg::init(int d, float mean, float var) {
    static std::default_random_engine e(time(0));
    static std::normal_distribution<float> n(mean, var);
    w = VectorXf::Zero(d+1, 1);
    // std::cout << "w:\n" << w.transpose() << std::endl;
    w.unaryExpr([](double dummy){return n(e);});
    // std::cout << "w:\n" << w.transpose() << std::endl;
}

VectorXf LogisticReg::predict(MatrixXf &X) {
    VectorXf a = X * w;
    VectorXf y(a);
    sigmoid(a, y);
    return y;
}

void LogisticReg::first_derivative(MatrixXf &X, VectorXf &y, VectorXf &dw) {
    int m = X.rows(), n = w.rows();
    VectorXf y_hat = predict(X);
    dw = VectorXf::Zero(n,1);
    VectorXf x_i(w);
    for(int i=0; i<m; ++i) {
        x_i = X.row(i).transpose();
        dw += x_i * (-y(i,0) + y_hat(i,0));
    }
}

void LogisticReg::second_derivative(MatrixXf &X, VectorXf &y, MatrixXf &d2w) {
    int m = X.rows(), n = w.rows();
    VectorXf y_hat = predict(X);
    // std::cout << "y_hat:\n" << y_hat << std::endl;
    d2w = MatrixXf::Zero(n, n);
    VectorXf x_i(w);
    for(int i=0; i<m; ++i) {
        x_i = X.row(i).transpose();
        d2w += x_i * x_i.transpose() * y_hat(i,0) * (1.0 - y_hat(i,0));
    }
}

void LogisticReg::solve(MatrixXf &X, VectorXf &y, int step=1000, float eps=1e-4) {
    VectorXf first_derv;
    MatrixXf second_derv;
    for(int i=0; i<step; ++i) {
        first_derivative(X, y, first_derv);
        // second_derivative(X, y, second_derv);
        // std::cout << first_derv.array().abs().sum() << std::endl;
        if(first_derv.array().abs().sum() < eps) {
            std::cout << i << std::endl;
            break;
        }
        // std::cout << "first_derv:\n" << first_derv << std::endl;
        // std::cout << "second_derv:\n" << second_derv << std::endl;
        GradDesc::step(w, first_derv);
    }
}

void sigmoid(VectorXf &x, VectorXf &y) {
    int n = x.rows();
    for(int i=0; i<n; ++i) {
        y(i, 0) = 1 / (1 + std::exp(-x(i, 0)));
    }
}