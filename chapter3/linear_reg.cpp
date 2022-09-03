#include "linear_reg.h"
#include <iostream>

VectorXf LinearReg::predict(const MatrixXf& X) {
    VectorXf pred = X * w;
    std::cout << pred.transpose() << std::endl;
    for(int i=0; i<X.rows(); ++i) {
        if(pred(i) >= 0.5) {
            pred(i) = 1;
        } else {
            pred(i) = 0;
        }
    }
    return pred;
}
