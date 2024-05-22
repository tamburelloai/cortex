//
// Created by Michael Tamburello on 5/21/24.
//

#include <iostream>
#include <algorithm>    // For std::shuffle
#include <fstream>      // For file operations
#include <sstream>      // For string stream operations
#include <random>       // For std::random_device and std::mt19937
#include <unordered_map>
#include <set>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <iostream>
#include "matrix.h"

using namespace std;


class OneHotEncoder {
private:
    std::unordered_map<std::string, int> labelToIdx;
public:
    void fit(const std::vector<string>& labels) {
        set<string> uniqueLabels = set<string>(labels.begin(), labels.end());
        int index = 0;
        for (auto it = uniqueLabels.begin(); it != uniqueLabels.end(); ++it) {
            labelToIdx[*it] = index;
            index++;
        }
        for (const auto& pair : labelToIdx) {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }

    }

    std::vector<double> _transform(std::string& label) {
        std::vector<double> res(labelToIdx.size(), 0.0); // init zero vector of size C (num classes)
        int idx = labelToIdx[label];
        res[idx] = 1.0;
        return res;
    }

    std::vector<std::vector<double>> transform(std::vector<std::string>& labels) {
        vector<vector<double>> result;
        for (auto label : labels) {
            std::vector<double> oneHotVec = _transform(label);
            result.push_back(oneHotVec);
        }
        return result;
    }

    std::vector<std::vector<double>> fit_transform(std::vector<std::string>& labels) {
        fit(labels);
        return transform(labels);
    }

};

class Dataset {
private:
    OneHotEncoder encoder;
    std::string dataset_path;
    std::vector<std::vector<double>> X;
    std::vector<std::string> original_y;
    std::vector<std::vector<double>> y;

public:
    Dataset() = default;
    Dataset(const std::string& filename) {
        dataset_path = filename;
        loadFromFile(filename);
        X = standardizeFeatures(X);
        y = encoder.fit_transform(original_y);
    }


    vector<vector<double>> standardizeFeatures(std::vector<std::vector<double>>& data) {
        size_t numRows = data.size();
        size_t numCols = data[0].size();

        for (size_t j = 0; j < numCols; ++j) {
            double mean = 0.0;
            double std_dev = 0.0;

            // Calculate mean
            for (size_t i = 0; i < numRows; ++i) {
                mean += data[i][j];
            }
            mean /= numRows;

            // Calculate standard deviation
            for (size_t i = 0; i < numRows; ++i) {
                std_dev += (data[i][j] - mean) * (data[i][j] - mean);
            }
            std_dev = std::sqrt(std_dev / numRows);

            // Normalize each element in the column
            for (size_t i = 0; i < numRows; ++i) {
                data[i][j] = (data[i][j] - mean) / (std_dev == 0 ? 1 : std_dev); // Avoid division by zero
            }
        }
        return data;
    }


    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        std::getline(file, line); //skips header (column names row)
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string value;
            std::vector<double> tmpX;
            std::string tmpy;
            int count = 0;
            while (getline(iss, value, ',')) {
                if (count < 4) {
                    tmpX.push_back(std::stod(value));
                } else {
                    tmpy = value;
                }
                count++;
            }
            X.push_back(tmpX);
            original_y.push_back(tmpy.substr(1, tmpy.size() - 2)); // removes quotes
        }
    }

    std::vector<std::vector<double>> getX() {
        return X;
    }

    std::pair<std::vector<double>, std::vector<double>> getSample(size_t idx) {
        vector<double> xSample = X.at(idx);
        vector<double> ySample = y.at(idx);
        return {xSample, ySample};
    }
};


class DataLoader {
private:
    Dataset ds;
    int batchSize;
    bool shuffle;
    vector<size_t> indices;
    size_t currentBatchIndex;

public:
    DataLoader(Dataset& ds, int batchSize, bool shuffle) {
        this->ds = ds;
        this->batchSize = batchSize;
        this->shuffle = shuffle;
        initialize();
    }

    void initialize() {
        indices.resize(ds.getX().size());
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle) {
            random_device rd;
            std::mt19937  g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }
        currentBatchIndex = 0;
    }

    std::pair<Matrix, Matrix> nextBatch() {
        vector<vector<double>> xBatch;
        vector<vector<double>> yBatch;
        size_t end = std::min(currentBatchIndex + batchSize, indices.size());
        for (size_t i = currentBatchIndex; i < end; i++) {
            pair<vector<double>, vector<double>> sample = ds.getSample(i);
            xBatch.push_back(sample.first);
            yBatch.push_back(sample.second);
        }
        currentBatchIndex = end;
        return {Matrix(xBatch), Matrix(yBatch)};
    }

    bool hasMoreBatches() const { return currentBatchIndex < indices.size(); }

    void reset() {
        currentBatchIndex = 0;
        if (shuffle) {
            std::random_device rd;  // Obtain a random number from hardware
            std::mt19937 g(rd());   // Seed the generator
            std::shuffle(indices.begin(), indices.end(), g);  // Shuffle the indices
        }
    }
};
