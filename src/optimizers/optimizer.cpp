//
// Created by Michael Tamburello on 5/20/24.
//

#include "optimizers/optimizer.h"


Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

void Optimizer::registerParameters(const std::vector<NN::Parameter*>& params) {
    for (auto param : params) {
        parameters.push_back(param);
    }
}