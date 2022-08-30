#include <memory>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "basekernel.h"

using std::vector;
using std::pair;
using std::shared_ptr;
using std::string;
using Eigen::VectorXf;
using Eigen::MatrixXf;

float clip(float, float, float);

int select_j(int, int);

class SVM {
private:
    VectorXf alphas;
    vector<pair<float, VectorXf>> support_vec;
    float b;
    float C;
    shared_ptr<BaseKernel> kernel;
    float _predict(const MatrixXf&, const VectorXf&, const VectorXf&);
public:
    SVM() : C(1.0) {}
    SVM(float _c, shared_ptr<KernelFactory> factory) : b(0), C(_c), 
                                                            kernel(factory->create()) { }
    void solve(const MatrixXf&, const VectorXf&, int);
    float predict(const VectorXf&);
    VectorXf predict(const MatrixXf &);
    inline VectorXf get_alphas() const { return alphas; }
    inline float get_b() const { return b; }
};


