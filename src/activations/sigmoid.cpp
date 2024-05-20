//
// Created by Michael Tamburello on 5/20/24.
//

#include "../include/activations/activation.h"
#include "../include/activations/sigmoid.h"

Matrix Sigmoid::forward(const Matrix &input) const {
    return input.apply([](double x) { return 1.0 / (1.0 + exp(-x)); });
}

Matrix Sigmoid::backward(const Matrix &input) const {
    Matrix sigmoidOutput = forward(input);
    return sigmoidOutput.apply([](double x) {return x * (1-x); });
}
