#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class LinearReg {
public:
    inline VectorXf predict(MatrixXf &X) {
        return X * w;
    }
    inline void solve(MatrixXf &X, VectorXf &y) {
        w = (X.transpose() * X).inverse() * X.transpose() * y;
    }
    inline VectorXf get_w() {
        return w;
    }

private:
    VectorXf w;
};

