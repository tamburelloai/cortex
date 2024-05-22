//
// Created by Michael Tamburello on 5/20/24.
//
#include "../include/data_utils/cortex_dataset.h"
#include <algorithm>    // For std::shuffle
#include <fstream>      // For file operations
#include <sstream>      // For string stream operations
#include <random>       // For std::random_device and std::mt19937
#include <unordered_map>
#include <set>


template<typename T, typename U>
CortexDataset<T, U>::CortexDataset() {}


template<typename T, typename U>
void CortexDataset<T, U>::loadFromFile(const std::string& filename, bool header) {
    std::ifstream file(filename);
    std::string line;

    if (header && std::getline(file, line)) {
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, ',')) {  // Assuming CSV format for headers
            headers.push_back(field);
        }
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        T input;
        U output;
        // You will need to adjust the extraction logic based on your data structure
        if (!(iss >> input >> output)) { break; } // Error handling or logging can be added here
        addData(input, output);
    }
}


template<typename T, typename U>
void CortexDataset<T, U>::addData(const T& input, const U& output) {
    data.emplace_back(input, output);
}

template<typename T, typename U>
std::pair<T, U>& CortexDataset<T, U>::get(size_t index) {
    return data[index];
}

template<typename T, typename U>
size_t CortexDataset<T, U>::size() const {
    return data.size();
}

template<typename T, typename U>
void CortexDataset<T, U>::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
}


template<typename T, typename U>
std::pair<CortexDataset<T, U>, CortexDataset<T, U>> CortexDataset<T, U>::split(float validation_fraction) {
    shuffle();  // Shuffle data before splitting
    size_t split_index = static_cast<size_t>(data.size() * (1 - validation_fraction));
    CortexDataset<T, U> train_set, validation_set;

    for (size_t i = 0; i < split_index; i++) {
        train_set.addData(data[i].first, data[i].second);
    }
    for (size_t i = split_index; i < data.size(); i++) {
        validation_set.addData(data[i].first, data[i].second);
    }

    return {train_set, validation_set};
}



template<typename T, typename U>
void CortexDataset<T, U>::encodeCategoricalColumn(size_t columnIndex) {
    std::set<T> unique_items;
    for (const auto& row : data) {
        unique_items.insert(row[columnIndex]);
    }
    int index = 0;
    for (const auto& item : unique_items) {
        category_encodings[columnIndex][item] = index++;
    }
    for (auto& row : data) {
        row[columnIndex] = static_cast<T>(category_encodings[columnIndex][row[columnIndex]]);
    }
}
