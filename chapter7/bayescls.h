#include <iostream>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <float.h>

#include "../ml.h"

#define PI 3.1415926

using std::map;
using std::set;
using std::pair;
using std::vector;
using std::unordered_map;

float gaussian(float x, float mean, float std);

pair<float,float> mean_std(const vector<float> &nums);

class NaiveBayes: public Classifier {
private:
    map<int, float> prior;
    map<int, map<int, map<int, float>>>    likelyhood1; // for sparse property
    map<int, map<int, pair<float, float>>> likelyhood2; // for continuous property
    vector<int> sparse_idx, dense_idx;
    map<int, int> type2num;
public:
    virtual void fit(const MatrixXf &X, const VectorXf &y) override;
    virtual VectorXf predict(const MatrixXf &X) override;
};