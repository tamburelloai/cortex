//
// Created by Michael Tamburello on 5/20/24.
//

#include "../include/activations/activation.h"
#include "../include/activations/softmax.h"
#include "../include/matrix.h"
#include <cmath>
#include <numeric>

Matrix Softmax::forward(const Matrix &input) const {
    Matrix result = input.apply([](double x) { return exp(x); });
    for (size_t i = 0; i < result.numRows(); i++) {
        double sum = 0;
        for (size_t j = 0; j < result.numCols(); j++) {
            sum += result(i, j);
        }
        for (size_t j = 0; j < result.numCols(); j++) {
            result(i, j) /= sum;
        }
    }

}

Matrix Softmax::backward(const Matrix &input) const {
    throw std::logic_error("Backward pass for Softmax should be handled"
                           "externally in the cross-entropy loss");
}

