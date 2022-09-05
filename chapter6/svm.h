#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "basekernel.h"
#include "../ml.h"

using std::vector;
using std::pair;
using std::shared_ptr;
using std::string;
using Eigen::VectorXf;
using Eigen::MatrixXf;

float clip(float, float, float);

int select_j(int, int);

class SVM: public Classifier{
private:
    VectorXf alphas;
    vector<pair<float, VectorXf>> support_vec;
    int max_step;
    float b;
    float C;
    shared_ptr<BaseKernel> kernel;
    float _predict(const MatrixXf&, const VectorXf&, const VectorXf&);
public:
    SVM() : b(0), C(1.0), max_step(10000) {}
    SVM(float _c, int _max_step, shared_ptr<KernelFactory> factory) : b(0), C(_c), max_step(_max_step), kernel(factory->create()) { }
    virtual void fit(const MatrixXf&, const VectorXf&) override;
    float predict(const VectorXf&);
    virtual VectorXf predict(const MatrixXf &) override;
    inline VectorXf get_alphas() const { return alphas; }
    inline float get_b() const { return b; }
};


