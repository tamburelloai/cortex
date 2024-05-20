//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_OPTIMIZER_H
#define CORTEX_OPTIMIZER_H
#include <vector>
#include <iostream>
#include "parameter.h"


class Optimizer {
protected:
    std::vector<NN::Parameter*> parameters;

public:
    Optimizer();
    virtual ~Optimizer();
    virtual void update() = 0;
    void registerParameters(const std::vector<NN::Parameter*>& params);
};


#endif //CORTEX_OPTIMIZER_H
