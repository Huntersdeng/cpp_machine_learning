#ifndef BASEKERNEL_H
#define BASEKERNEL_H
#include <Eigen/Core>
#include <memory>
#include <vector>

using Eigen::VectorXf;
using Eigen::MatrixXf;

using std::shared_ptr;
using std::vector;

class BaseKernel {
public:
    BaseKernel() {}
    virtual float operator() (const VectorXf &x1, const VectorXf& x2) = 0;
    virtual VectorXf operator() (const MatrixXf &x1, const VectorXf& x2) = 0;
    virtual ~BaseKernel() { }
};

class KernelFactory {
public:
    KernelFactory() {}
    KernelFactory(vector<float> _p) : param(_p) {}
    virtual shared_ptr<BaseKernel> create()=0;
    virtual ~KernelFactory() {}
protected:
    vector<float> param;
};

#endif