//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_CROSSENTROPY_LOSS_H
#define CORTEX_CROSSENTROPY_LOSS_H

#include "loss.h"
#include "matrix.h"
#include <cmath>  // Include cmath for the log function

namespace NN {
    class CategoricalCrossEntropyLoss {
    public:
        Matrix yhat;
        double compute(const Matrix& logits, const Matrix& actual_labels)  {
            int N = logits.numRows();
            int K = logits.numCols();
            yhat = Matrix(N, K);
            double L = 0.0;

            //compute softmax
            for (size_t i = 0; i < N; ++i) {
                double max_logit = *std::max_element(logits.data[i].begin(), logits.data[i].end()); // for numerical stability
                double denominator = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    yhat.data[i][k] = std::exp(logits.data[i][k] - max_logit); // shift by max logit for stability
                    denominator += yhat.data[i][k];
                }
                for (size_t k = 0; k < K; ++k) {
                    yhat.data[i][k] /= denominator; // Normalize to make probabilities sum to 1
                }
            }
            // Compute cross-entropy loss
            for (size_t i = 0; i < N; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    L += actual_labels.data[i][k] * std::log(yhat.data[i][k] + 1e-9); // add small number to prevent log(0)
                }
            }
            L = (-1.0 / N) * L; // Average loss over the batch
            return L;
        }

        Matrix backward(const Matrix& actual_labels) {
            int N = yhat.numRows();
            int K = yhat.numCols();
            Matrix dL_dz = Matrix(N, K); // Derivative of loss with respect to logits

            // Compute gradient of loss w.r.t. logits
            for (size_t i = 0; i < N; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    dL_dz.data[i][k] = yhat.data[i][k] - actual_labels.data[i][k];
                }
            }
            return dL_dz;
        }
    };
}

#endif //CORTEX_CROSSENTROPY_LOSS_H
