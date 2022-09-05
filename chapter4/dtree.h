#include <vector>
#include <map>
#include <memory>
#include <Eigen/Core>

#include "../ml.h"

using std::vector;
using std::map;
using std::shared_ptr;
using Eigen::MatrixXf;
using Eigen::VectorXf;

enum principle {ENT, GINI};

float entropy(const VectorXf &y, const vector<int> &idx);

float entropy_gain(const MatrixXf &X, const VectorXf &y, 
                                const vector<int> &idx, int property_id);

float gain_ratio(const MatrixXf &X, const VectorXf &y, 
                                const vector<int> &idx, int property_id);

float gini(const VectorXf &y, const vector<int> &idx);

float gini_index(const MatrixXf &X, const VectorXf &y, 
                                const vector<int> &idx, int property_id);

bool is_equal_property(const MatrixXf&, const vector<int>&, const vector<int>&);

int best_property(const MatrixXf &X, const VectorXf &y, 
                  const vector<int> &idx, const vector<int> &properties, int method);

int majority_cls(const VectorXf &y, const vector<int> &idx);

class DNode {
public:
    int cls;
    int property_id;
    map<int, std::shared_ptr<DNode>> children;
    DNode() : cls(-1), property_id(-1) {}
};

class DTree : public Classifier{
public:
    DTree(int _m) : method(_m) {}
    virtual void fit(const MatrixXf &X, const VectorXf &y) override;
    virtual VectorXf predict(const MatrixXf &X) override;
private:
    shared_ptr<DNode> tree_generate(const MatrixXf &X, const VectorXf &y, 
                                    const vector<int> &idx, vector<int> &properties);
    shared_ptr<DNode> root;
    int method;
};