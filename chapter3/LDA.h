#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using Eigen::VectorXf;
using Eigen::MatrixXf;

void covariance(MatrixXf &X, MatrixXf &cov);

class LDA{
private:
    float x0, x1;
    VectorXf w;
public:
    LDA(int d) : x0(0), x1(0), w(VectorXf::Zero(d, 1)) {}
    void solve(MatrixXf &X, VectorXf &y);
    VectorXf predict(MatrixXf &X);
};

void covariance(MatrixXf &X, MatrixXf &cov) {
    MatrixXf meanVec = X.colwise().mean();
    Eigen::RowVectorXf meanVecRow(Eigen::RowVectorXf::Map(meanVec.data(),X.cols()));

    Eigen::MatrixXf zeroMeanMat = X;
    zeroMeanMat.rowwise() -= meanVecRow;
    if(X.rows()==1)
        cov = (zeroMeanMat.adjoint()*zeroMeanMat)/double(X.rows());
    else
        cov = (zeroMeanMat.adjoint()*zeroMeanMat)/double(X.rows()-1);
}

void LDA::solve(MatrixXf &X, VectorXf &y) {
    int m = X.rows(), d = X.cols();
    int m0 = 0, m1 = 0;
    for(int i=0; i<m; ++i) {
        if(y(i, 0) == 0) ++m0;
        else             ++m1;
    }
    MatrixXf X0, X1;
    X0.resize(m0, d);
    X1.resize(m1, d);
    m0 = m1 = 0;
    for(int i=0; i<m; ++i) {
        if(y(i, 0) == 0) {
            X0.row(m0++) = X.row(i);
        }
        else {
            X1.row(m1++) = X.row(i);
        }
    }
    VectorXf mean0 = X0.colwise().mean();
    VectorXf mean1 = X1.colwise().mean();
    MatrixXf var0, var1;
    covariance(X0, var0);
    covariance(X1, var1);

    MatrixXf Sw = var0 + var1;
    std::cout << Sw.rows() << " " << Sw.cols() << std::endl;
    std::cout << mean0 << std::endl;
    w = Sw.inverse() * (mean0 - mean1);
    x0 = w.transpose() * mean0;
    x1 = w.transpose() * mean1;
}

VectorXf LDA::predict(MatrixXf &X) {
    int m = X.rows();
    VectorXf y_hat = X * w;
    VectorXf pred(y_hat);
    for(int i=0; i<m; ++i) {
        float distance1 = abs(x0 - y_hat(i, 0));
        float distance2 = abs(x1 - y_hat(i, 0));
        pred(i, 0) = distance1 < distance2 ? 0 : 1;
    }
    return pred;
}