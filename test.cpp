#include <iostream>
#include "utils/equation.h"
#include <Eigen/Core>

using std::cout;
using std::endl;

int main() {
    Eigen::Matrix2f M;
    M << 2, 3, 4, 5; 
    Eigen::Vector2f vec = M.row(0);
    cout << M << endl;
    cout << vec << endl;
}