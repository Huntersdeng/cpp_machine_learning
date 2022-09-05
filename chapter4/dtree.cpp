#include "dtree.h"

#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <float.h>
#include <memory>
#include <algorithm>
#include <functional>

using std::vector;
using std::map;
using std::set;
using std::shared_ptr;

float entropy(const VectorXf &y, const vector<int> &idx) {
    float ent = 0.0;
    int count0 = 0, count1 = 0, sum = idx.size();
    for(int i : idx) {
        if(y(i) == 0) ++count0;
        else          ++count1;
    }

    if(count0 > 0) {
        float p0 = count0 / (float)sum;
        ent += - p0 * log(p0) / log(2);
    }
    if(count1 > 0) {
        float p1 = count1 / (float)sum;
        ent += - p1 * log(p1) / log(2);
    }    
    return ent;
}

float entropy_gain(const MatrixXf &X, const VectorXf &y, 
                   const vector<int> &idx, int property_id) {
    set<int> property;
    for(int i : idx) {
        property.insert(X(i,property_id));
    }
    float gain = entropy(y, idx);
    for(int p : property) {
        vector<int> sub_idx;
        for(int i : idx) {
            if(X(i,property_id) == p) {
                sub_idx.push_back(i);
            }
        }
        gain -= sub_idx.size() / (float)idx.size() * entropy(y, sub_idx);
    }
    return gain;
}

float gain_ratio(const MatrixXf &X, const VectorXf &y, 
                 const vector<int> &idx, int property_id) {
    set<int> property;
    for(int i : idx) {
        property.insert(X(i,property_id));
    }
    float gain = entropy(y, idx);
    float IV = 0.0, ratio = 0.0;
    for(int p : property) {
        vector<int> sub_idx;
        for(int i : idx) {
            if(X(i,property_id) == p) {
                sub_idx.push_back(i);
            }
        }
        ratio = sub_idx.size() / (float)idx.size();
        gain -= ratio * entropy(y, sub_idx);
        IV -= ratio * log(ratio) / log(2);
    }
    return gain / IV;
}

float gini(const VectorXf &y, const vector<int> &idx) {
    int count0 = 0, count1 = 0;
    for(int i : idx) {
        if(y(i) == 0) ++count0;
        else          ++count1;
    }
    float p0 = count0 / (float)idx.size();
    float p1 = count1 / (float)idx.size();
    return 1.0 - p0 * p0 - p1 * p1;
}

float gini_index(const MatrixXf &X, const VectorXf &y, 
                 const vector<int> &idx, int property_id) {
    set<int> property;
    for(int i : idx) {
        property.insert(X(i,property_id));
    }
    float val = 0.0;
    for(int p : property) {
        vector<int> sub_idx;
        for(int i : idx) {
            if(X(i,property_id) == p) {
                sub_idx.push_back(i);
            }
        }
        val -= sub_idx.size() / (float)idx.size() * entropy(y, sub_idx);
    }
    return val;
}

bool is_equal_property(const MatrixXf &X, 
                       const vector<int> &idx, 
                       const vector<int>& properties) {
    if(idx.size() == 1) {
        return true;
    }
    const VectorXf &x = X.row(idx[0]);
    for(int i : idx) {
        for(int p : properties) {
            if(X(i,p) != x(p)) {
                return false;
            }
        }
    }
    return true;
}

int best_property(const MatrixXf &X, const VectorXf &y, 
                          const vector<int> &idx, const vector<int> &properties, int method) {
    float max_purity = -FLT_MAX;
    int best_p = -1;
    std::function<float(const MatrixXf&, const VectorXf&, 
                        const vector<int>&, int)> func;
    switch (method) {
        case principle::ENT: func = entropy_gain; break;
        case principle::GINI:   func = gini_index;   break;
    }

    for(int p : properties) {
        float purity = func(X, y, idx, p);
        if(max_purity < purity) {
            max_purity = purity;
            best_p = p;
        }
    }
    return best_p;
}

int majority_cls(const VectorXf &y, const vector<int> &idx) {
    int count0 = 0, count1 = 0;
    for(int i : idx) {
        if(y(i) == 0) ++count0;
        else          ++count1;
    }
    return count0 > count1 ? 0 : 1;
}

shared_ptr<DNode> DTree::tree_generate(const MatrixXf &X, const VectorXf &y, 
                                       const vector<int> &idx, vector<int> &properties) {
    /* 1. 生成节点 */
    shared_ptr<DNode> node = std::make_shared<DNode>();

    /* 2. 判断是否属于同一类别 */
    set<int> cls_set;
    for(int i : idx) {
        cls_set.insert(y(i));
    }
    if(cls_set.size() == 1) {
        node->cls = y(idx[0]);
        return node;
    }

    /* 3. 判断是否为 空 或 特征取值全部相同 */
    if(properties.size() == 0 or is_equal_property(X, idx, properties)) {
        node->cls = majority_cls(y, idx);
        return node;
    }

    /* 4. 选择最优划分属性 */
    int property_id = best_property(X, y, idx, properties, method);
    auto it = std::find(properties.begin(), properties.end(), property_id);
    properties.erase(it);

    /* 5. 根据划分属性的取值，生成子节点 */
    node->property_id = property_id;
    set<int> property_values;
    for(int i : idx) {
        property_values.insert(X(i,property_id));
    }
    for(int val : property_values) {
        vector<int> sub_idx;
        for(int i : idx) {
            if(X(i,property_id) == val) {
                sub_idx.push_back(i);
            }
        }
        if(sub_idx.size() == 0) { 
            node->children[val] = std::make_shared<DNode>();
            node->children[val]->cls = majority_cls(y, idx);
        } else {
            node->children[val] = tree_generate(X, y, sub_idx, properties);
        }
    }
    return node;
}

void DTree::fit(const MatrixXf &X, const VectorXf &y) {
    vector<int> idx, properties;
    for(int i=0; i<X.rows(); ++i) {
        idx.push_back(i);
    }
    for(int j=0; j<X.cols(); ++j) {
        properties.push_back(j);
    }
    root = tree_generate(X, y, idx, properties);
}

VectorXf DTree::predict(const MatrixXf &X) {
    int m = X.rows();
    VectorXf pred(m);
    for(int i=0; i<X.rows(); ++i) {
        shared_ptr<DNode> node = root;
        VectorXf x = X.row(i);
        while(node->cls == -1) {
            node = node->children[x(node->property_id)];
        }
        pred(i) = node->cls;
    }
    return pred;
}