#include "dataset.h"

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

void read_csv2matrix(const string &path, MatrixXf &X, VectorXf &y, bool verpose) {
    vector<vector<float>> X_data;
    vector<int> y_data;
    read_csv2vector(path, X_data, y_data);
    int m = X_data.size(), d = X_data[0].size();
    X.resize(m, d);
    y.resize(m, 1);
    for(int i=0; i<m; ++i) {
        for(int j=0; j<d+1; ++j) {
            if(j == d) {
                y(i, 0) = y_data[i];
            } else {
                X(i, j) = X_data[i][j];
            }
        }
    }
    if(verpose) {
        std::cout << X << std::endl;
        std::cout << y.transpose() << std::endl;
    }
}

int reverse_int(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_mnist_label(string filename, VectorXf&labels, int label0, int label1, vector<int> &idx) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;

        vector<int> vec_labels;
        int n = 0;
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
            if((float)label == label0) {
                vec_labels.push_back(0);
                idx.push_back(i);
                ++n;
            } else if((float)label == label1) {
                vec_labels.push_back(1);
                idx.push_back(i);
                ++n;
            }
		}

        labels.resize(vec_labels.size());
        for(int i=0; i<vec_labels.size(); ++i) {
            labels(i) = vec_labels[i];
        }
	}
}

void read_mnist_images(string filename, MatrixXf&images, const vector<int> idx) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		n_rows = reverse_int(n_rows);
		n_cols = reverse_int(n_cols);
 
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;
        
        int n = 0;
        images.resize(idx.size(), n_rows*n_cols);
		for (int i = 0; i < number_of_images; i++)
		{   
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
                    if(i == idx[n]) images(n, r*n_cols+c) = image;
				}
			}
            if(i == idx[n]) ++n;
		}
	}
}