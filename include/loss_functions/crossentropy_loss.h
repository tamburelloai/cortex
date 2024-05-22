//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_CROSSENTROPY_LOSS_H
#define CORTEX_CROSSENTROPY_LOSS_H

#include "loss.h"
#include "matrix.h"
#include <cmath>  // Include cmath for the log function

namespace NN {
    class CategoricalCrossEntropyLoss: public Loss {
    public:
        double compute(const Matrix& predicted, const Matrix& actual) override {
            double loss = 0.0;
            for (size_t i = 0; i < predicted.numRows(); ++i) {
                for (size_t j = 0; j < predicted.numCols(); ++j) {
                    loss -= actual(i, j) * std::log(std::max(predicted(i, j), 1e-9));  // Using max to ensure non-negative input to log
                }
            }
            return loss / predicted.numRows();  // Average loss per batch
        }


        Matrix gradient(const Matrix& predicted, const Matrix& actual) override {
            Matrix grad = predicted;  // Start with copying predicted
            for (size_t i = 0; i < predicted.numRows(); ++i) {
                for (size_t j = 0; j < predicted.numCols(); ++j) {
                    grad(i, j) = predicted(i, j) - actual(i, j);  // Simplified gradient for softmax with cross-entropy
                }
            }
            return grad;
        }
    };
}

#endif //CORTEX_CROSSENTROPY_LOSS_H
