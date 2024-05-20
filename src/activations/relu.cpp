//
// Created by Michael Tamburello on 5/20/24.
//

#include "../include/activations/activation.h"
#include "../include/activations/relu.h"
#include <algorithm>

Matrix ReLU::forward(const Matrix& input) const {
    return input.apply([](double x) { return std::max(0.0, x); });
}

Matrix ReLU::backward(const Matrix& input) const {
    return input.apply([](double x) {return x > 0 ? 1.0 : 0.0;});
}




