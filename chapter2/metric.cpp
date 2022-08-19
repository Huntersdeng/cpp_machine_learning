#include "metric.h"

#include <iostream>
#include <cmath>
#include <Eigen/Core>

using Eigen::VectorXf;
using Eigen::Matrix2i;

float Metric::mse(const VectorXf &pred, const VectorXf &label) {
    assert(pred.rows() == label.rows());
    return (pred - label).squaredNorm();
}

float Metric::accuracy(const VectorXf &pred, const VectorXf &label) {
    assert(pred.rows() == label.rows());
    int sum = pred.rows();
    float count = 0;
    for(int i=0; i<sum; ++i) {
        if(abs(pred(i, 0) - label(i, 0))<0.0001) {
            ++count;
        }
    }
    return count / sum;
}

float Metric::error_rate(const VectorXf &pred, const VectorXf &label) {
    return 1.0 - accuracy(pred, label);
}

void Metric::confusion_matrix(const VectorXf &pred, const VectorXf &label, Matrix2i &mat) {
    assert(pred.rows() == label.rows());
    int TP=0, FN=0, FP=0, TN=0;
    int n = pred.rows();
    for(int i=0; i<n; ++i) {
        if(pred(i, 0) == 1) {
            if(label(i, 0) == 1) ++TP;
            else                 ++FP;
        } else {
            if(label(i, 0) == 1) ++FN;
            else                 ++TN;
        }
    }
    mat << TP, FN, FP, TN;
}

float Metric::precision(const VectorXf &pred, const VectorXf &label) {
    Matrix2i mat;
    confusion_matrix(pred, label, mat);
    float TP = mat(0, 0), FP = mat(1, 0);
    return TP / (TP + FP);
}

float Metric::recall(const VectorXf &pred, const VectorXf &label) {
    Matrix2i mat;
    confusion_matrix(pred, label, mat);
    float TP = mat(0, 0), FN = mat(0, 1);
    return TP / (TP + FN);
}

float Metric::f1_score(const VectorXf &pred, const VectorXf &label) {
    Matrix2i mat;
    confusion_matrix(pred, label, mat);
    float TP = mat(0, 0), FN = mat(0, 1), FP = mat(1, 0);
    return 2 * TP / (2 * TP + FP + FN);
}

float Metric::f_beta_score(const VectorXf &pred, const VectorXf &label, float beta) {
    Matrix2i mat;
    confusion_matrix(pred, label, mat);
    float TP = mat(0, 0), FN = mat(0, 1), FP = mat(1, 0);
    float a = (1 + beta * beta) * TP;
    float b = (TP + FP + beta * beta * (TP + FN));
    return a / b;
}

float Metric::jaccard_coefficient(const VectorXf &pred, const VectorXf &label) {
    int a=0, b=0, c=0, d=0;
    int m = label.rows();
    for(int i=0; i<m; ++i) {
        for(int j=i+1; j<m; ++j) {
            if(pred(i) == pred(j)) {
                if(label(i) == label(j)) ++a;
                else                     ++b;
            } else {
                if(label(i) == label(j)) ++c;
                else                     ++d;
            }
        }
    }
    return a / (float)(a+b+c);
}

float Metric::fmi(const VectorXf &pred, const VectorXf &label) {
    int a=0, b=0, c=0, d=0;
    int m = label.rows();
    for(int i=0; i<m; ++i) {
        for(int j=i+1; j<m; ++j) {
            if(pred(i) == pred(j)) {
                if(label(i) == label(j)) ++a;
                else                     ++b;
            } else {
                if(label(i) == label(j)) ++c;
                else                     ++d;
            }
        }
    }
    float index = a * a / (float)(a+b) / (float)(a+c);
    return sqrt(index);
}