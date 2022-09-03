#include <Eigen/Core>

using Eigen::MatrixXf;
using Eigen::VectorXf;

class LVQ{
private:
    MatrixXf P;
public:
    LVQ() { }
    void solve(const MatrixXf&, const VectorXf&, float, int);
    VectorXf predict(const MatrixXf&);
};