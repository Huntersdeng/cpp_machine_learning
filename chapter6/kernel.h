#include <Eigen/Core>

using Eigen::VectorXf;
using Eigen::MatrixXf;

class BaseKernel {
public:
    BaseKernel() {}
    virtual float operator() (const VectorXf &x1, const VectorXf& x2) = 0;
};

class LinearKernel: public BaseKernel {
public:
    inline float operator() (const VectorXf &x1, const VectorXf& x2) override{
        return x1.transpose() * x2;
    }
};

class PolyKernel: public BaseKernel {
public:
    PolyKernel() : d(2) {}
    PolyKernel(int _d) : d(_d) { assert(_d >= 1); }
    inline float operator() (const VectorXf &x1, const VectorXf& x2) override{
        return pow(x1.transpose() * x2, d);
    }
private:
    int d;
};

class RBFKernel: public BaseKernel {
public:
    RBFKernel() : sigma(1.0) {}
    RBFKernel(float _sigma) : sigma(_sigma) {}
    float operator() (const VectorXf &x1, const VectorXf& x2) override;
private:
    float sigma;
};

class LaplaceKernel: public BaseKernel {
public:
    LaplaceKernel() : sigma(1.0) {}
    LaplaceKernel(float _sigma) : sigma(_sigma) { assert(_sigma > 0); }
    float operator() (const VectorXf &x1, const VectorXf& x2) override;
private:
    float sigma;
};

class SigmoidKernel: public BaseKernel {
public:
    SigmoidKernel() : beta(1.0), theta(0.0) {}
    SigmoidKernel(float _beta, float _theta) : beta(_beta), theta(_theta) {
        assert(_beta > 0);
        assert(_theta < 0);
    }
    float operator() (const VectorXf &x1, const VectorXf& x2) override;
private:
    float beta;
    float theta;
};