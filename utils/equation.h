#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

double grad_desc(int a, int b, int c, float lr = 0.0001, int step = 1000) {
    srand((int)(time(NULL)));
    double grad = 1.0, x = rand() / double(RAND_MAX);
    int i = 0;
    while(i < step) {
        grad = 2 * (a * x * x + b * x + c) * (2 * a * x + b);
        x = x - lr * grad;
        ++i;
    }
    return x;
}

double newton_iter(int a, int b, int c, int step=1000) {
    srand((int)(time(NULL)));
    double y, k, d;
    double x = rand() / double(RAND_MAX);
    int i = 0;
    while(i < step) {
        y = a * x * x + b * x + c;
        k = 2 * a * x + b;
        d = y - k * x;
        x = - d / k;
        ++i;
    }
}