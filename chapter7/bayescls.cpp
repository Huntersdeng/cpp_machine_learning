#include "bayescls.h"

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

void NaiveBayes::fit(const MatrixXf &X, const VectorXf &y) {
    for(int i=0; i<y.rows(); ++i) {
        ++type2num[y(i)];
    }

    /* 1. 计算先验概率 */
    for(auto &kv : type2num) {
        prior[kv.first] = (kv.second+1) / (float)(y.rows() + type2num.size());
    }

    /* 2. 判断属性是离散值还是连续值 */
    map<int, set<float>> val_range;
    for(int i=0; i<X.cols(); ++i) {
        set<float> range;
        for(int j=0; j<X.rows(); ++j) {
            range.insert(X(j,i));
        }
        val_range[i] = set<float>(range);
        if(2 * range.size() < y.rows()) {
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
    for(int i=0; i<X.rows(); ++i) {
        for(int j : sparse_idx) {
            ++likelyhood1[y(i)][j][X(i,j)];
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
        for(int i=0; i<X.rows(); ++i) {
            vals[y(i)].push_back(X(i,j));
        }
        for(auto &kv : vals) {
            likelyhood2[kv.first][j] = pair<float,float>(mean_std(kv.second));
        }
    }
}

VectorXf NaiveBayes::predict(const MatrixXf &X) {
    int m = X.rows();
    VectorXf pred(m);
    set<int> types;
    for(auto &kv : type2num) {
        types.insert(kv.first);
    }
    for(int i=0; i<m; ++i) {
        map<int, float> log_likelyhood;
        for(int j : sparse_idx) {
            for(int type: types) {
                log_likelyhood[type] += log(likelyhood1[type][j][X(i,j)]);
            }
        }
        for(int j : dense_idx) {
            for(int type: types) {
                pair<float, float> m_s = likelyhood2[type][j];
                log_likelyhood[type] += log(gaussian(X(i,j), m_s.first, m_s.second));
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
        pred(i) = best_type;
    }
    return pred;
}