#include "lvq.h"
#include <float.h>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <Eigen/Core>

using std::vector;
using std::map;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::RowVectorXf;

void LVQ::solve(const MatrixXf &X, const VectorXf &y, float lr, int max_iter) {
    /* 1. 初始化原型向量P */
    map<int, vector<int>> T;
    for(int i=0; i<y.rows(); ++i) {
        T[y(i)].push_back(i);
    }
    int m = X.rows(), q = T.size();
    P.resize(q, X.cols());
    for(int i=0; i<q; ++i) {
        P.row(i) = X.row(T[i][0]);
    }

    /* 2. 循环 */

    std::default_random_engine e(time(0));
    std::uniform_int_distribution<int> u(0, m-1);

    int iter = 0;
    while(iter < max_iter) {
        int j = u(e);
        int i_star = -1;
        float min_dist = FLT_MAX;
        for(int i=0; i<q; ++i) {
            float dist = (X.row(j) - P.row(i)).norm();
            if(dist < min_dist) {
                min_dist = dist;
                i_star = i;
            }
        }
        if(y(j) == i_star) {
            P.row(i_star) = lr * X.row(j) + (1 - lr) * P.row(i_star);
        } else {
            P.row(i_star) = -lr * X.row(j) + (1 + lr) * P.row(i_star);
        }
        ++iter;
    }
}

VectorXf LVQ::predict(const MatrixXf&X) {
    int m = X.rows(), q = P.rows();
    VectorXf pred(m);
    for(int j=0; j<m; ++j) {
        int i_star = -1;
        float min_dist = FLT_MAX;
        for(int i=0; i<q; ++i) {
            float dist = (X.row(j) - P.row(i)).norm();
            if(dist < min_dist) {
                min_dist = dist;
                i_star = i;
            }
        }
        pred(j) = i_star;
    }
    return pred;
}