//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_LAYER_H
#define CORTEX_LAYER_H

#include "matrix.h"
#include "parameter.h"
#include <vector>

namespace NN {
    class Layer {
    public:
        virtual ~Layer() = default;
        virtual Matrix forward(const Matrix& input) = 0;
        virtual Matrix backward(const Matrix& gradOutput) = 0;
        virtual std::vector<Parameter*> params() {
            return std::vector<Parameter*>();
        }
    };
}

#endif // LAYER_H


