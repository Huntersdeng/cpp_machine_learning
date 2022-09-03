#include <vector>
#include <Eigen/Core>

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowVectorXf;



class KMeans {
public:
    KMeans(int _k) : k(_k), clusters(_k) {}
    void solve(const MatrixXf&, VectorXf &);
private:
    int k;
    vector<vector<int>> clusters;
};