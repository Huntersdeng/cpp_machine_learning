#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Core>

using std::vector;
using std::string;
using Eigen::MatrixXf;
using Eigen::VectorXf;

void read_csv2vector(const string&, vector<vector<float>>&, vector<int>&);
void read_csv2matrix(const string&, MatrixXf&, VectorXf&, bool);

void read_csv2vector(const string& path, vector<vector<float>> &X, vector<int> &y) {
    std::ifstream file(path);
    std::cout << "Reading data from " + path + "...  " << std::endl;
    string line;
    while(std::getline(file, line)) {
        string element;
        vector<float> row;
        std::istringstream ss(line);
        std::getline(ss, element, '\t');
        while(std::getline(ss, element, '\t')) {
            row.push_back(std::stof(element));
        }
        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }
}

void read_csv2matrix(const string &path, MatrixXf &X, VectorXf &y, bool verpose=true) {
    vector<vector<float>> X_data;
    vector<int> y_data;
    read_csv2vector(path, X_data, y_data);
    int m = X_data.size(), d = X_data[0].size();
    X.resize(m, d+1);
    y.resize(m, 1);
    for(int i=0; i<m; ++i) {
        for(int j=0; j<d+1; ++j) {
            if(j == d) {
                y(i, 0) = y_data[i];
                X(i, j) = 1;
            } else {
                X(i, j) = X_data[i][j];
            }
        }
    }
    if(verpose) {
        std::cout << X << std::endl;
        std::cout << y << std::endl;
    }
}