//
// Created by Michael Tamburello on 5/21/24.
//
#include "../include/activations/relu.h"

namespace NN {

    ReLU::ReLU() {
        // Constructor may be empty if no initialization is needed
    }

    Matrix ReLU::forward(const Matrix& input) {
        this->input = input;  // Store input for use in backward pass
        output = input.apply([](float x) { return std::max(0.0f, x); });
        return output;
    }

    Matrix ReLU::backward(const Matrix& gradOutput) {
        Matrix gradientInput = input.apply([](float x) { return x > 0 ? 1.0f : 0.0f; });
        // Assume you need to multiply gradOutput by gradientInput to propagate the gradient
        return gradientInput * gradOutput; // This is a placeholder; adjust as necessary
    }

}
