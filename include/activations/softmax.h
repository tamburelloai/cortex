//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_SOFTMAX_H
#define CORTEX_SOFTMAX_H


#include "../include/layers/layer.h"
#include "../matrix.h"

namespace NN {
    class Softmax : public Layer {
    private:
        Matrix input;
        Matrix output;

    public:
        Softmax();
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradOutput) override;

        // Additional methods if needed
    };
}

#endif //CORTEX_SOFTMAX_H
