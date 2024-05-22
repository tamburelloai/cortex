//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_RELU_H
#define CORTEX_RELU_H

#include "../include/layers/layer.h"
#include "../matrix.h"

namespace NN {
    class ReLU : public Layer {
    private:
        Matrix input;
        Matrix output;

    public:
        ReLU();
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradOutput) override;

        // Additional methods if needed
    };
}

#endif //CORTEX_RELU_H
