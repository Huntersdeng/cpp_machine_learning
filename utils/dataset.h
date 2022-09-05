#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>

using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::fstream;
using Eigen::MatrixXf;
using Eigen::VectorXf;

void read_csv2vector(const string&, vector<vector<float>>&, vector<int>&);
void read_csv2matrix(const string&, MatrixXf&, VectorXf&, bool=true);
int reverse_int(int i);
void read_mnist_label(string filename, VectorXf&labels, int label0, int label1, vector<int> &idx);
void read_mnist_images(string filename, MatrixXf&images, const vector<int> idx);

#endif
 
// int main()
// {
// 	/*
// 	vector<double>labels;
// 	read_Mnist_Label("t10k-labels.idx1-ubyte", labels);
// 	for (auto iter = labels.begin(); iter != labels.end(); iter++)
// 	{
// 		cout << *iter << " ";
// 	}
// 	*/
// 	vector<vector<double>>images;
// 	read_Mnist_Images("t10k-images.idx3-ubyte", images);
// 	for (int i = 0; i < images.size(); i++)
// 	{
// 		for (int j = 0; j < images[0].size(); j++)
// 		{
// 			cout << images[i][j] << " ";
// 		}
// 	}
// 	return 0;
// }