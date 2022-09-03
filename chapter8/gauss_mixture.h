#include <Eigen/Core>
#include <vector>

using Eigen::MatrixXf;
using Eigen::VectorXf;

using std::vector;

#define PI 3.1415926

class GaussianMixture{
public:
    GaussianMixture(int _k) : k(_k), alphas(_k), variances(_k), betas(_k) {}
    void solve(const MatrixXf&, int);
    VectorXf predict(const MatrixXf&);
private:
    void init(int);
    int k;
    vector<float> alphas;
    MatrixXf means;
    vector<MatrixXf> variances;
    vector<MatrixXf> betas;
};

static float gaussian(const VectorXf& x, const VectorXf& mean, const MatrixXf& beta);