#include <Eigen/Core>

#include "../ml.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class LVQ : public Classifier{
private:
    MatrixXf P;
    float lr;
    int max_step;
public:
    LVQ() : lr(0.0001), max_step(10000) { }
    LVQ(float _lr, int _max_step) : lr(_lr), max_step(_max_step) {}
    virtual void fit(const MatrixXf&, const VectorXf&) override;
    virtual VectorXf predict(const MatrixXf&) override;
};