//
// Created by Michael Tamburello on 5/22/24.
//
// Adam.h
// Adam.h
#ifndef CORTEX_ADAM_H
#define CORTEX_ADAM_H

#include "optimizer.h"
#include <cmath>
#include <map>

namespace Optim {
    class Adam : public Optimizer {
    private:
        float learningRate;
        float beta1;
        float beta2;
        float epsilon;
        int timestep;  // To keep track of the number of updates

        // Maps to store the first and second moments of the gradients for each parameter
        std::map<NN::Parameter*, Matrix> m;
        std::map<NN::Parameter*, Matrix> v;

    public:
        Adam(std::vector<NN::Parameter*>& params, float lr, float b1 = 0.9, float b2 = 0.999, float eps = 1e-8)
                : Optimizer(params), learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0) {}

        void step() override {
            timestep++;
            for (auto param : parameters) {
                // Initialize moments if not already done
                if (m.find(param) == m.end()) {
                    m[param] = Matrix(param->data.numRows(), param->data.numCols());
                    v[param] = Matrix(param->data.numRows(), param->data.numCols());
                }

                // Update first and second moments and parameters
                for (size_t i = 0; i < param->data.numRows(); ++i) {
                    for (size_t j = 0; j < param->data.numCols(); ++j) {
                        float grad_ij = param->grad(i, j);

                        // Update first moment (mean of gradients)
                        m[param](i, j) = beta1 * m[param](i, j) + (1 - beta1) * grad_ij;

                        // Update second moment (uncentered variance of gradients)
                        v[param](i, j) = beta2 * v[param](i, j) + (1 - beta2) * (grad_ij * grad_ij);

                        // Compute bias-corrected first moment estimate
                        float m_hat = m[param](i, j) / (1 - pow(beta1, timestep));

                        // Compute bias-corrected second moment estimate
                        float v_hat = v[param](i, j) / (1 - pow(beta2, timestep));

                        // Update parameter
                        param->data(i, j) -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
                    }
                }
            }
        }
    };
}

#endif // CORTEX_ADAM_H
