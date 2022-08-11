#include <Eigen/Core>

using Eigen::VectorXf;
using Eigen::MatrixXf;

float clip(float, float, float);

int select_j(int, int);

class SVM {
private:
    VectorXf w;
    float b;
    float C;
    void update_w(const VectorXf&, const MatrixXf&, const VectorXf&);
public:
    SVM(float _c) : C(_c) {}
    void solve(const MatrixXf&, const VectorXf&, int);
    float predict(const VectorXf&);
    VectorXf predict(const MatrixXf &);
    inline VectorXf get_w() { return w; }
    inline float get_b() { return b; }
};


