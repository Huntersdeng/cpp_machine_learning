#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../ml.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;

class LinearReg : public Classifier{
public:
    virtual VectorXf predict(const MatrixXf &X) override;
    virtual inline void fit(const MatrixXf &X, const VectorXf &y) override{
        w = (X.transpose() * X).inverse() * X.transpose() * y;
    }
    inline VectorXf get_w() {
        return w;
    }

private:
    VectorXf w;
};

