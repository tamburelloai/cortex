//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_SGD_H
#define CORTEX_SGD_H

#include "optimizer.h"

namespace Optim {
    class SGD : public Optimizer {
    private:
        float learningRate;
    public:
        SGD(std::vector<NN::Parameter*>& params, float lr)
        : Optimizer(params), learningRate(lr) {}

        void step() override {
            for (auto param : parameters) {
                param->data = param->data  - ( learningRate * param->grad );
            }
        }

    };
}

#endif //CORTEX_SGD_H
