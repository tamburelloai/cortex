//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_CORTEX_DATASET_H
#define CORTEX_CORTEX_DATASET_H

#include <vector>
#include <utility>
#include <string>
#include <unordered_map>

using namespace std;

template<typename T, typename U>
class CortexDataset {
private:
    vector<pair<T, U>> data;
    std::vector<std::string> headers;   // Stores column headers if any
    std::unordered_map<size_t, std::unordered_map<T, int>> category_encodings; // Maps column index to a map of category to encoding
    std::vector<U> labels;



public:
    CortexDataset();
    virtual ~CortexDataset() {}

    void loadFromFile(const std::string& filename, bool header = false);

    // Add a single data point
    void addData(const T& input, const U& output);

    // Get data point
    std::pair<T, U>& get(size_t index);

    // Get total number of data points
    size_t size() const;

    // Shuffle the data
    void shuffle();

    // train/val split
    pair<CortexDataset, CortexDataset> split(float validation_fraction);

    void encodeCategoricalColumn(size_t columnIndex);


};

#endif //CORTEX_CORTEX_DATASET_H
