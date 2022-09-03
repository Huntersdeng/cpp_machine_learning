#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "../ml.h"
#include "../utils/solver.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;

void sigmoid(const VectorXf &input, VectorXf &output);

class LogisticReg : public Classifier{
public:
    LogisticReg(int _max_step=10000, float _eps=1e-4) : max_step(_max_step), eps(_eps){}
    virtual VectorXf predict(const MatrixXf&) override;
    virtual void fit(const MatrixXf&, const VectorXf&) override;
    inline VectorXf get_w() { return w; }
private:
    VectorXf _predict(const MatrixXf&);
    void first_derivative(const MatrixXf &X, const VectorXf &y, VectorXf &dw);
    void second_derivative(const MatrixXf &X, const VectorXf &y, MatrixXf &d2w);
    void init(int d, float mean=0.0, float var=10.0);
    VectorXf w;
    int max_step;
    float eps;
};