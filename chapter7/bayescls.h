#include <iostream>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <float.h>

#define PI 3.1415926

using std::map;
using std::set;
using std::pair;
using std::vector;
using std::unordered_map;

float gaussian(float x, float mean, float std);

pair<float,float> mean_std(const vector<float> &nums);

class NaiveBayes {
private:
    map<int, float> prior;
    map<int, map<int, map<int, float>>>    likelyhood1; // for sparse property
    map<int, map<int, pair<float, float>>> likelyhood2; // for continuous property
    vector<int> sparse_idx, dense_idx;
    map<int, int> type2num;
public:
    void solve(const vector<vector<float>> &X, const vector<int> &y);
    vector<int> predict(const vector<vector<float>> &X);
};

float gaussian(float x, float mean, float std) {
    float norm_x = (x - mean) / std;
    return exp(-norm_x * norm_x / 2)  /  (sqrt(2*PI) * std);
}

pair<float,float> mean_std(const vector<float> &nums) {
    float sum = std::accumulate(std::begin(nums), std::end(nums), 0.0);  
    float mean = sum / (float)nums.size();
    float accum  = 0.0;  
    std::for_each (std::begin(nums), std::end(nums), [&](const float d) {  
        accum  += (d-mean)*(d-mean);  
    });  

    float stdev = sqrt(accum/(nums.size()-1));
    return pair<float,float>(mean, stdev);
}

void NaiveBayes::solve(const vector<vector<float>> &X, const vector<int> &y) {
    for(int i=0; i<y.size(); ++i) {
        ++type2num[y[i]];
    }

    /* 1. 计算先验概率 */
    for(auto &kv : type2num) {
        prior[kv.first] = (kv.second+1) / (float)(y.size() + type2num.size());
    }

    /* 2. 判断属性是离散值还是连续值 */
    map<int, set<float>> val_range;
    for(int i=0; i<X[0].size(); ++i) {
        set<float> range;
        for(int j=0; j<X.size(); ++j) {
            range.insert(X[j][i]);
        }
        val_range[i] = set<float>(range);
        if(2 * range.size() < y.size()) {
            sparse_idx.push_back(i);
        } else {
            dense_idx.push_back(i);
        }
    }
    
    /* 3. 计算离散属性的似然 */
    for(auto &t_n : type2num) {
        for(auto &k_v : val_range) {
            for(float v : k_v.second) {
                likelyhood1[t_n.first][k_v.first][v] = 0;
            }
            
        }
    }
    for(int i=0; i<X.size(); ++i) {
        for(int j : sparse_idx) {
            ++likelyhood1[y[i]][j][X[i][j]];
        }
    }
    for(auto &kv1 : likelyhood1) {
        int type = kv1.first;
        for(auto &kv2 : kv1.second) {
            int prop = kv2.first;
            for(auto &kv3 : kv2.second) {
                int val = kv3.first;
                float cnt = kv3.second;
                kv3.second = (cnt + 1) / (float)(type2num[type] + val_range[prop].size());
            }
        }
    }

    /* 4. 计算连续属性的似然 */
    map<int, vector<float>> vals;
    for(int j : dense_idx) {
        vals.clear();
        for(int i=0; i<X.size(); ++i) {
            vals[y[i]].push_back(X[i][j]);
        }
        for(auto &kv : vals) {
            likelyhood2[kv.first][j] = pair<float,float>(mean_std(kv.second));
        }
    }
}

vector<int> NaiveBayes::predict(const vector<vector<float>> &X) {
    vector<int> pred;
    set<int> types;
    for(auto &kv : type2num) {
        types.insert(kv.first);
    }
    for(int i=0; i<X.size(); ++i) {
        map<int, float> log_likelyhood;
        for(int j : sparse_idx) {
            for(int type: types) {
                log_likelyhood[type] += log(likelyhood1[type][j][X[i][j]]);
            }
        }
        for(int j : dense_idx) {
            for(int type: types) {
                pair<float, float> m_s = likelyhood2[type][j];
                log_likelyhood[type] += log(gaussian(X[i][j], m_s.first, m_s.second));
            }
        }
        float max_log_likelyhood=-FLT_MAX;
        int best_type = -1;
        for(auto &kv : log_likelyhood) {
            if(max_log_likelyhood < kv.second) {
                best_type = kv.first;
                max_log_likelyhood = kv.second;
            }
        }
        pred.push_back(best_type);
    }
    return pred;
}