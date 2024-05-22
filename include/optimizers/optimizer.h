//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_OPTIMIZER_H
#define CORTEX_OPTIMIZER_H
#include <vector>
#include "../include/parameter.h"

namespace Optim {
class Optimizer {
public:
    Optimizer(std::vector<NN::Parameter*>& params)
    : parameters(params) {}

    virtual ~Optimizer() {}
    virtual void step() = 0;

protected:
    std::vector<NN::Parameter*>& parameters;
};

}

#endif // OPTIMIZER_H