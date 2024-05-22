//
// Created by Michael Tamburello on 5/21/24.
//

#include "../include/activations/softmax.h"

namespace NN {

    Softmax::Softmax() {
        // Constructor may be empty if no initialization is needed
    }

    Matrix Softmax::forward(const Matrix& input) {
        this->input = input; // Store the original input to use in the backward pass if needed
        output = Matrix(input.rows(), input.cols()); // Initialize the output matrix with the same dimensions as input

        for (int i = 0; i < input.rows(); ++i) {
            // Apply the softmax to each row individually
            double maxVal = *std::max_element(input.row(i).begin(), input.row(i).end()); // Stability trick: get the maximum value in the row

            // Calculate the sum of exp(values - maxVal) for normalization
            double sumExp = 0.0;
            for (int j = 0; j < input.cols(); ++j) {
                double expVal = exp(input(i, j) - maxVal); // Exponentiate each element minus the max value for stability
                output(i, j) = expVal;
                sumExp += expVal;
            }

            // Normalize each element by the sum of all exponentiated values
            for (int j = 0; j < input.cols(); ++j) {
                output(i, j) /= sumExp;
            }
        }

        return output;
    }



    Matrix Softmax::backward(const Matrix& target) {
        // Assuming 'output' holds the softmax probabilities computed in the forward pass
        // and 'target' is the one-hot encoded true labels
        Matrix gradInput = output;  // Start with the softmax output
        gradInput = gradInput - target;        // Subtract the target (one-hot true labels)
        return gradInput;
    }


}
