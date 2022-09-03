#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../ml.h"

using Eigen::VectorXf;
using Eigen::MatrixXf;

void covariance(const MatrixXf &X, MatrixXf &cov);

class LDA: public Classifier{
private:
    float x0, x1;
    VectorXf w;
    inline void init(int d) {
        w = VectorXf::Zero(d, 1);
    }
public:
    LDA() : x0(0), x1(0) {}
    virtual void fit(const MatrixXf &X, const VectorXf &y) override;
    virtual VectorXf predict(const MatrixXf &X) override;
    virtual ~LDA() {}
};