#include "kmeans.h"
#include <Eigen/Core>
#include <algorithm>
#include <float.h>

using std::vector;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowVectorXf;

void KMeans::fit(const MatrixXf &X) {    
    int m = X.rows(), d = X.cols();
    /* 1. 随机选择数据集中k个向量作为起始簇中心 */
    means.resize(k, d);
    srand(unsigned(time(NULL)));
    vector<int> randomVec(m, 0);
    for(int i = 0; i < m; ++i)
    {
        randomVec[i] = i;
    }
    std::random_shuffle(randomVec.begin(), randomVec.end());

    for(int i=0; i<k; ++i) {
        means.row(i) = X.row(randomVec[i]);
    }
    
    /* 2. 迭代 */
    vector<vector<int>> clusters(k);
    bool flag = true;
    while(flag) {
        for(int i=0; i<k; ++i) {
            clusters[i].clear();
        }
        flag = false;
        for(int j=0; j<m; ++j) {
            float min_dist = FLT_MAX;
            int cluster_idx = -1;
            for(int i=0; i<k; ++i) {
                float dist = (X.row(j) - means.row(i)).norm();
                if(dist < min_dist) {
                    min_dist = dist;
                    cluster_idx = i;
                }
            }
            clusters[cluster_idx].push_back(j);
        }
        for(int i=0; i<k; ++i) {
            RowVectorXf mean = VectorXf::Zero(X.cols());
            for(int j : clusters[i]) {
                mean += X.row(j);
            }
            mean /= clusters[i].size();
            if((mean - means.row(i)).norm() > 0.00001) {
                means.row(i) = mean;
                flag = true;
            }
        }
    }
}

VectorXf KMeans::predict(const MatrixXf &X) {
    int m = X.rows();
    VectorXf pred(m);
    for(int j=0; j<m; ++j) {
        float min_dist = FLT_MAX;
        int cluster_idx = -1;
        for(int i=0; i<k; ++i) {
            float dist = (X.row(j) - means.row(i)).norm();
            if(dist < min_dist) {
                min_dist = dist;
                cluster_idx = i;
            }
        }
        pred(j) = cluster_idx;
    }
    return pred;
}