//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_DATALOADER_H
#define CORTEX_DATALOADER_H

#include "cortex_dataset.h"
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
using namespace std;

template<typename T, typename U>
class DataLoader {
private:
    CortexDataset<T, U>& dataset;
    size_t batchSize;
    bool shuffle;
    vector<size_t> indices;
    size_t currentBatchIndex;
public:
    DataLoader(CortexDataset<T, U>& dataset, size_t batchSize, bool shuffle = true);
    void initialize();
    vector<pair<T, U>> nextBatch();

    bool hasMoreBatches() const;
    void reset();

};
#endif //CORTEX_DATALOADER_H
