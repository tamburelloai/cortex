//
// Created by Michael Tamburello on 5/20/24.
//

#include "../include/data_utils/dataloader.h"

template<typename T, typename U>
DataLoader<T, U>::DataLoader(CortexDataset<T, U>& dataset, size_t batchSize, bool shuffle)
        : dataset(dataset), batchSize(batchSize), shuffle(shuffle), currentBatchIndex(0) {
    initialize();
}

template<typename T, typename U>
void DataLoader<T, U>::initialize() {
    indices.resize(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);  // Fill indices with 0, 1, 2, ..., n

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
}

template<typename T, typename U>
std::vector<std::pair<T, U>> DataLoader<T, U>::nextBatch() {
    std::vector<std::pair<T, U>> batch;
    size_t start = currentBatchIndex * batchSize;
    size_t end = std::min(start + batchSize, dataset.size());

    for (size_t i = start; i < end; ++i) {
        batch.push_back(dataset.get(indices[i]));
    }

    currentBatchIndex++;
    return batch;
}

template<typename T, typename U>
bool DataLoader<T, U>::hasMoreBatches() const {
    return currentBatchIndex * batchSize < dataset.size();
}

template<typename T, typename U>
void DataLoader<T, U>::reset() {
    currentBatchIndex = 0;
    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
}
