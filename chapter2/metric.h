#include <iostream>
#include <Eigen/Core>

using Eigen::VectorXf;
using Eigen::Matrix2i;

class Metric {
public:
    static float mse(const VectorXf &pred, const VectorXf &label);
    static float accuracy(const VectorXf &pred, const VectorXf &label);
    static float error_rate(const VectorXf &pred, const VectorXf &label);
    static void confusion_matrix(const VectorXf &pred, const VectorXf &label, Matrix2i &mat);
    static float precision(const VectorXf &pred, const VectorXf &label);
    static float recall(const VectorXf &pred, const VectorXf &label);
    static float f1_score(const VectorXf &pred, const VectorXf &label);
    static float f_beta_score(const VectorXf &pred, const VectorXf &label, float beta);
};