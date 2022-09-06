#include "gauss_mixture.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::VectorXf;

using std::vector;

float gaussian(const VectorXf& x, const VectorXf& mean, const MatrixXf& beta) {
    int n = mean.cols();
    float divisor = sqrt(beta.determinant() / pow(2*PI, n));
    float val = -0.5 * ((x - mean).transpose() * beta * (x - mean))(0,0);
    return exp(val) / divisor;
}

void GaussianMixture::init(int cols) {
    for(int i=0; i<k; ++i) {
        alphas[i] = 1 / (float)k;
    }
    means = MatrixXf::Random(k, cols);
    for(int i=0; i<k; ++i) {
        variances[i] = MatrixXf::Identity(cols, cols);
        betas[i] = MatrixXf::Identity(cols, cols);
    }
}

void GaussianMixture::fit(const MatrixXf& X) {
    int m = X.rows(), n = X.cols();
    init(n);
    // std::cout << "means:\n" << means << std::endl;
    // std::cout << "variances:\n";
    // for(int i=0; i<n; ++i){
    //     std::cout << i << "\n" << variances[i] << std::endl;
    // }
    int iter = 0;
    MatrixXf gamma(m, k);
    while(iter < max_step) {
        for(int j=0; j<m; ++j) {
            float norm = 0.0;
            for(int i=0; i<k; ++i) {
                gamma(j, i) = alphas[i] * gaussian(X.row(j), means.row(i), betas[i]);
                norm += gamma(j, i);
            }
            for(int i=0; i<k; ++i) {
                gamma(j, i) /= norm;
            }
        }
        // std::cout << gamma << std::endl;
        for(int i=0; i<k; ++i) {
            float norm = gamma.col(i).sum();
            means.row(i) = gamma.col(i).transpose() * X / norm;
            MatrixXf var = MatrixXf::Zero(n, n);
            for(int j=0; j<m; ++j) {
                var += gamma(j, i) * (X.row(j) - means.row(i)).transpose() * (X.row(j) - means.row(i));
            }
            variances[i] = var / norm;
            alphas[i] = norm / m;
            betas[i] = variances[i].inverse();
        }
        // std::cout << "means:\n" << means << std::endl;
        // std::cout << "variances:\n";
        // for(int i=0; i<n; ++i){
        //     std::cout << i << "\n" << variances[i] << std::endl;
        // }
        ++iter;
    }
}

VectorXf GaussianMixture::predict(const MatrixXf &X) {
    int m = X.rows();
    MatrixXf gamma(m, k);
    for(int j=0; j<m; ++j) {
        float norm = 0.0;
        for(int i=0; i<k; ++i) {
            gamma(j, i) = alphas[i] * gaussian(X.row(j), means.row(i), betas[i]);
            norm += gamma(j, i);
        }
        for(int i=0; i<k; ++i) {
            gamma(j, i) /= norm;
        }
    }
    VectorXf pred(m);
    for(int j = 0; j < m; ++j) {
        int idx = -1;
        float max_prob = 0;
        for(int i=0; i<k; ++i) {
            // std::cout << "gamma(j,i):" << gamma(j, i) << std::endl;
            if(max_prob < gamma(j, i)) {
                max_prob = gamma(j, i);
                idx = i;
            }
        }
        pred(j) = idx;
    }
    return pred;
}