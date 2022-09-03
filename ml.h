#ifndef ML_H
#define ML_H

#include <Eigen/Core>

using Eigen::VectorXf;
using Eigen::MatrixXf;

class Classifier{
public:
    virtual void fit(const MatrixXf &X, const VectorXf &y)=0;
    virtual VectorXf predict(const MatrixXf &X)=0;
    virtual ~Classifier() {}
};

#endif