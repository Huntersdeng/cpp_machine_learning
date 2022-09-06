#include <vector>
#include <Eigen/Core>
#include "../ml.h"

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowVectorXf;



class KMeans : public Cluster{
public:
    KMeans(int _k) : k(_k) {}
    virtual void fit(const MatrixXf &X) override;
    virtual VectorXf predict(const MatrixXf &X) override;
private:
    int k;
    MatrixXf means;
};