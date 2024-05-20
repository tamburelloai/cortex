//
// Created by Michael Tamburello on 5/19/24.
//

#include "../../include/layers/linear.h"

using namespace NN;

Linear::Linear(size_t inputSize, size_t outputSize, InitType initType)
    : weights(inputSize, outputSize, initType),
      bias(1, outputSize, InitType::Zeros) {}


Matrix Linear::forward(const Matrix& input) {
    this->input = input; // Store input for use in backward pass
    // Implement the forward logic correctly
    output = this->input.matMul(weights.getData()) + bias.getData();
    return output;
}

void Linear::backward() {

}