#include "logistic_reg.h"

void LogisticReg::init(int d, float mean, float var) {
    static std::default_random_engine e(time(0));
    static std::normal_distribution<float> n(mean, var);
    w = VectorXf::Zero(d+1, 1);
    // std::cout << "w:\n" << w.transpose() << std::endl;
    w.unaryExpr([](double dummy){return n(e);});
    // std::cout << "w:\n" << w.transpose() << std::endl;
}

void LogisticReg::fit(const MatrixXf &X, const VectorXf &y){
    int m = X.rows(), d = X.cols();
    init(d);
    MatrixXf X_hat(X);
    X_hat.conservativeResize(m, d + 1);
    X_hat.col(d) = VectorXf::Ones(m);
    VectorXf first_derv;
    MatrixXf second_derv;
    for(int i=0; i<max_step; ++i) {
        first_derivative(X_hat, y, first_derv);
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

VectorXf LogisticReg::predict(const MatrixXf &X){
    int m = X.rows(), d = X.cols();
    MatrixXf X_hat(X);
    X_hat.conservativeResize(m, d + 1);
    X_hat.col(d) = VectorXf::Ones(m);
    VectorXf a = X_hat * w;
    VectorXf y(a);
    sigmoid(a, y);
    for(int i=0; i<X.rows(); ++i) {
        if(y(i) >= 0.5) {
            y(i) = 1;
        } else {
            y(i) = 0;
        }
    }
    return y;
}

VectorXf LogisticReg::_predict(const MatrixXf &X){
    VectorXf a = X * w;
    VectorXf y(a);
    sigmoid(a, y);
    return y;
}

void LogisticReg::first_derivative(const MatrixXf &X, const VectorXf &y, VectorXf &dw) {
    int m = X.rows(), n = w.rows();
    VectorXf y_hat = _predict(X);
    dw = VectorXf::Zero(n,1);
    VectorXf x_i(w);
    for(int i=0; i<m; ++i) {
        x_i = X.row(i).transpose();
        dw += x_i * (-y(i,0) + y_hat(i,0));
    }
}

void LogisticReg::second_derivative(const MatrixXf &X, const VectorXf &y, MatrixXf &d2w) {
    int m = X.rows(), n = w.rows();
    VectorXf y_hat = _predict(X);
    // std::cout << "y_hat:\n" << y_hat << std::endl;
    d2w = MatrixXf::Zero(n, n);
    VectorXf x_i(w);
    for(int i=0; i<m; ++i) {
        x_i = X.row(i).transpose();
        d2w += x_i * x_i.transpose() * y_hat(i,0) * (1.0 - y_hat(i,0));
    }
}

void sigmoid(const VectorXf &input, VectorXf &output) {
    int n = input.rows();
    for(int i=0; i<n; ++i) {
        output(i) = 1 / (1 + std::exp(-input(i)));
    }
}