#include <Eigen/Core>
#include <memory>
#include <vector>

#include "basekernel.h"

using Eigen::VectorXf;
using Eigen::MatrixXf;

using std::shared_ptr;
using std::vector;

class LinearKernel: public BaseKernel {
public:
    inline float operator() (const VectorXf &x1, const VectorXf& x2) override{
        return x1.transpose() * x2;
    }
    VectorXf operator() (const MatrixXf &x1, const VectorXf& x2) override;
};

class LinearKernelFactory: public KernelFactory {
public:
    LinearKernelFactory() {}
    LinearKernelFactory(vector<float> param) : KernelFactory(param) {}
    virtual shared_ptr<BaseKernel> create();
};

class PolyKernel: public BaseKernel {
public:
    PolyKernel() : d(2) {}
    PolyKernel(int _d) : d(_d) { assert(_d >= 1); }
    inline float operator() (const VectorXf &x1, const VectorXf& x2) override{
        return pow(x1.transpose() * x2, d);
    }
    VectorXf operator() (const MatrixXf &x1, const VectorXf& x2) override;
private:
    int d;
};

class PolyKernelFactory: public KernelFactory {
public:
    PolyKernelFactory() {}
    PolyKernelFactory(vector<float> param) : KernelFactory(param) {}
    virtual shared_ptr<BaseKernel> create();
};

class RBFKernel: public BaseKernel {
public:
    RBFKernel() : sigma(1.0) {}
    RBFKernel(float _sigma) : sigma(_sigma) { assert(_sigma > 0); }
    float operator() (const VectorXf &x1, const VectorXf& x2) override;
    VectorXf operator() (const MatrixXf &x1, const VectorXf& x2) override;
private:
    float sigma;
};

class RBFKernelFactory: public KernelFactory {
public:
    RBFKernelFactory() {}
    RBFKernelFactory(vector<float> param) : KernelFactory(param) {}
    virtual shared_ptr<BaseKernel> create();
};

class LaplaceKernel: public BaseKernel {
public:
    LaplaceKernel() : sigma(1.0) {}
    LaplaceKernel(float _sigma) : sigma(_sigma) { assert(_sigma > 0); }
    float operator() (const VectorXf &x1, const VectorXf& x2) override;
    VectorXf operator() (const MatrixXf &x1, const VectorXf& x2) override;
private:
    float sigma;
};

class LaplaceKernelFactory: public KernelFactory {
public:
    LaplaceKernelFactory() {}
    LaplaceKernelFactory(vector<float> param) : KernelFactory(param) {}
    virtual shared_ptr<BaseKernel> create();
};

class SigmoidKernel: public BaseKernel {
public:
    SigmoidKernel() : beta(1.0), theta(0.0) {}
    SigmoidKernel(float _beta, float _theta) : beta(_beta), theta(_theta) {
        assert(_beta > 0);
        assert(_theta < 0);
    }
    float operator() (const VectorXf &x1, const VectorXf& x2) override;
    VectorXf operator() (const MatrixXf &x1, const VectorXf& x2) override;
private:
    float beta;
    float theta;
};

class SigmoidKernelFactory: public KernelFactory {
public:
    SigmoidKernelFactory() {}
    SigmoidKernelFactory(vector<float> param) : KernelFactory(param) {}
    virtual shared_ptr<BaseKernel> create();
};