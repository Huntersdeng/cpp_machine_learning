#include "svm.h"
#include <iostream>
#include <memory>
#include <string>
#include <ctime>
#include <random>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>

using std::shared_ptr;
using std::string;
using Eigen::VectorXf;
using Eigen::MatrixXf;
using Eigen::RowVectorXf;

float clip(float alpha, float L, float H) {
    if (alpha < L) {
        return L;
    } else if (alpha > H) {
        return H;
    } else {
        return alpha;
    }
}

int select_j(int i, int m) {
    static std::default_random_engine e(time(0));
    static std::uniform_int_distribution<int> u(0,m-2);
    int res = u(e);
    if(res >= i) {
        ++res;
    }
    return res;
}

// void SVM::update_w(const VectorXf &alphas, const MatrixXf &dataset, const VectorXf &labels) {
//     int m = dataset.rows(), n = dataset.cols();
//     VectorXf ay(m);
//     for(int i=0; i<m; ++i) {
//         ay(i) = alphas(i, 0) * labels(i, 0);
//     }
//     w = dataset.transpose() * ay;
//     std::cout << "w:\n" << w.transpose() << std::endl;
// }

void SVM::fit(const MatrixXf &dataset, const VectorXf &labels) {
    /*  
    简化版SMO算法实现，未使用启发式方法对alpha对进行选择.
    :param dataset: 所有特征数据向量
    :param labels: 所有的数据标签
    :param C: 软间隔常数, 0 <= alpha_i <= C
    :param max_iter: 外层循环最大迭代次数
    */

    /* 1. 初始化变量 */
    int m = dataset.rows(), n = dataset.cols();
    alphas = VectorXf::Ones(m);
    VectorXf x_i, x_j;
    float a_i, b_i, y_i, fx_i, E_i;
    float a_j, b_j, y_j, fx_j, E_j;
    float K_ii, K_jj, K_ij, eta;
    float a_i_old, a_i_new, a_j_old, a_j_new;
    float L, H;
    int iter = 0;
    
    /* 2. 循环更新alphas与w，b */
    while(iter < max_step) {
        int j = 0;
        int pair_changed = 0;
        for(int i=0; i<m; ++i) {
            a_i = alphas(i, 0);
            x_i = dataset.row(i);
            y_i = labels(i,0);
            fx_i = _predict(dataset, labels, x_i);
            E_i = fx_i - y_i;

            j = select_j(i, m);
            a_j = alphas(j, 0);
            x_j = dataset.row(j);
            y_j = labels(j, 0);
            fx_j = _predict(dataset, labels, x_j);
            E_j = fx_j - y_j;

            K_ii = (*kernel)(x_i, x_i);
            K_jj = (*kernel)(x_j, x_j);
            K_ij = (*kernel)(x_i, x_j);

            eta = K_ii + K_jj - 2 * K_ij;
            if(eta <= 0) {
                std::cout << "WARNING: eta<=0" << std::endl;
                continue;
            }

            // 获取更新的alpha对
            a_i_old = a_i, a_j_old = a_j;
            a_j_new = a_j_old + y_j*(E_i - E_j) / eta;

            // 对alpha进行修剪
            if(y_i != y_j) {
                L = std::max<float>(0.0, a_j_old - a_i_old);
                H = std::min<float>(C, C + a_j_old - a_i_old);
            } else {
                L = std::max<float>(0.0, a_i_old + a_j_old - C);
                H = std::min<float>(C, a_j_old + a_i_old);
            }
            a_j_new = clip(a_j_new, L, H);
            a_i_new = a_i_old + y_i*y_j*(a_j_old - a_j_new);
            if(std::abs(a_j_new - a_j_old) < 0.00001) {
                continue;
            }
            alphas(i, 0) = a_i_new;
            alphas(j, 0) = a_j_new;

            // 更新阈值b
            b_i = -E_i - y_i*K_ii*(a_i_new - a_i_old) - y_j*K_ij*(a_j_new - a_j_old) + b;
            b_j = -E_j - y_i*K_ij*(a_i_new - a_i_old) - y_j*K_jj*(a_j_new - a_j_old) + b;
            if(a_i_new > 0 && a_i_new < C) {
                b = b_i;
            } else if(a_j_new > 0 && a_j_new < C) {
                b = b_j;
            } else {
                b = (b_i + b_j) / 2;
            }
            pair_changed += 1;
            printf("INFO   iteration:%d  i:%d  pair_changed:%d\n", iter, i, pair_changed);
        }
        if(pair_changed == 0) {
            ++iter;
        } else {
            iter = 0;
        }
        printf("iteration number: %d\n", iter);
    }
    std::cout << "alphas: " << alphas.transpose() << std::endl;

    /* 3. 保存支持向量 */
    for(int i=0; i<m; ++i) {
        if(abs(alphas(i)) > 0.001) {
            support_vec.emplace_back(alphas(i) * labels(i), dataset.row(i));
        }
    }
}

float SVM::_predict(const MatrixXf &dataset, const VectorXf &labels, const VectorXf &x) {
    int m = dataset.rows(), n = dataset.cols();
    // VectorXf K(m), ay(m);
    // for(int i=0; i<m; ++i) {
    //     K(i) = (*kernel)(dataset.row(i), x);
    //     ay(i) = alphas(i) * labels(i);
    // }
    VectorXf K = (*kernel)(dataset, x);
    VectorXf ay = alphas.array() * labels.array();
    return (ay.transpose() * K) + b;
}

VectorXf SVM::predict(const MatrixXf &X) {
    int m = support_vec.size(), n = X.rows();
    MatrixXf K(m, n);
    VectorXf ay(m);
    for(int i=0; i<m; ++i) {
        ay(i) = support_vec[i].first;
        for(int j=0; j<n; ++j) {
            K(i, j) = (*kernel)(support_vec[i].second, X.row(j));
        }
    }
    return (ay.transpose() * K).array() + b;
}

float SVM::predict(const VectorXf &x) {
    int m = support_vec.size();
    VectorXf K(m), ay(m);
    for(int i=0; i<m; ++i) {
        ay(i) = support_vec[i].first;
        K(i) = (*kernel)(support_vec[i].second, x);
    }
    return (ay.transpose() * K) + b;
}