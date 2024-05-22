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
        double compute(const Matrix& predicted_prob, const Matrix& actual_labels) override {
            int N = predicted_prob.numRows();
            int K = predicted_prob.numCols();
            double L = 0.0;
            double instance_sum = 0.0; // exterior sum (i=1...N)
            for (size_t i = 0; i < N; ++i) {
                double class_sum = 0.0;    // interior sum (k=1...K)
                for (size_t k = 0; k < K; ++k) {
                    class_sum += actual_labels(i, k) * std::log(std::max(predicted_prob(i, k), 1e-9));  // Using max to ensure non-negative input to log
                }
                instance_sum += class_sum;
            }
            L = -(1/N) * instance_sum;
            return L;
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
