#include <Eigen/Core>
#include <vector>
#include "../ml.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

using std::vector;

#define PI 3.1415926

class GaussianMixture : public Cluster{
public:
    GaussianMixture(int _k, int _max_step) : k(_k), alphas(_k), variances(_k), betas(_k), max_step(_max_step) {}
    virtual void fit(const MatrixXf&) override;
    virtual VectorXf predict(const MatrixXf&) override;
private:
    void init(int);
    int k;
    int max_step;
    vector<float> alphas;
    MatrixXf means;
    vector<MatrixXf> variances;
    vector<MatrixXf> betas;
};

static float gaussian(const VectorXf& x, const VectorXf& mean, const MatrixXf& beta);