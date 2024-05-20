//
// Created by Michael Tamburello on 5/20/24.
//

#include "../include/activations/tanh.h"

Matrix Tanh::forward(const Matrix &input) const {
    return input.apply([](double x) { return tanh(x); });
}

Matrix Tanh::backward(const Matrix &input) const {
    Matrix tanhOutput = forward(input);
    return tanhOutput.apply([](double x) { return 1 - x * x; });
}
