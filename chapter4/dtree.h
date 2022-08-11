#include <vector>
#include <map>
#include <memory>

using std::vector;
using std::map;
using std::shared_ptr;

#define ENTROPY_GAIN    0
#define GINI_INDEX      1

float entropy(const vector<int> &y, const vector<int> &idx);

float entropy_gain(const vector<vector<float>> &X, const vector<int> &y, 
                                const vector<int> &idx, int property_id);

float gain_ratio(const vector<vector<float>> &X, const vector<int> &y, 
                                const vector<int> &idx, int property_id);

float gini(const vector<int> &y, const vector<int> &idx);

float gini_index(const vector<vector<float>> &X, const vector<int> &y, 
                                const vector<int> &idx, int property_id);

bool is_equal_property(const vector<vector<float>>&, const vector<int>&, const vector<int>&);

int best_property(const vector<vector<float>> &X, const vector<int> &y, 
                  const vector<int> &idx, const vector<int> &properties, int method);

int majority_cls(const vector<int> &y, const vector<int> &idx);

class DNode {
public:
    int cls;
    int property_id;
    map<int, std::shared_ptr<DNode>> children;
    DNode() : cls(-1), property_id(-1) {}
};

class DTree{
public:
    DTree(int _m) : method(_m) {}
    void solve(const vector<vector<float>> &X, const vector<int> &y);
    vector<int> predict(const vector<vector<float>> &X);
private:
    shared_ptr<DNode> tree_generate(const vector<vector<float>> &X, const vector<int> &y, 
                         const vector<int> &idx, vector<int> &properties);
    int _predict(const vector<float> &x);
    shared_ptr<DNode> root;
    int method;
};