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
    output = this->input.matMul(weights.getData());
    output = output + bias.getData();
    return output;
}

Matrix Linear::backward(const Matrix& gradOutput) {
    Matrix gradWeights = input.transpose().matMul(gradOutput);
    weights.grad = gradWeights;
    //TODO Matrix gradBias = gradOutput.sumRows();
    //TODO bias.grad = gradBias;
    Matrix gradInput = gradOutput.matMul(weights.data.transpose());
    return gradInput;
}

std::vector<Parameter*> Linear::params() {
    return std::vector<Parameter*>{&weights, &bias};
}