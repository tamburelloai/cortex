//
// Created by Michael Tamburello on 5/20/24.
//

#include "optimizers/adam.h"
#include <cmath>

Adam::Adam(double lr, double b1, double b2, double eps)
    : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {

}

void Adam::update() {
    t++;
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& grad = parameters[i]->getGrad();  // Reference to avoid unnecessary copies
        m[i] = m[i] * beta1 + grad * (1 - beta1);  // Update first moment estimate
        v[i] = v[i] * beta2 + grad.apply([](double x) { return x * x; }) * (1 - beta2);  // Update second moment estimate

        auto m_hat = m[i] / (1 - std::pow(beta1, t));  // Bias-corrected first moment
        auto v_hat = v[i] / (1 - std::pow(beta2, t));  // Bias-corrected second moment

        // Adjusting learning rate for each element
        for (size_t row = 0; row < parameters[i]->getData().numRows(); row++) {
            for (size_t col = 0; col < parameters[i]->getData().numCols(); col++) {
                double adjustedLearningRate = (learningRate * m_hat(row, col)) / (sqrt(v_hat.getElement(row, col)) + epsilon);
                parameters[i]->getData()(row, col) -= adjustedLearningRate;
            }
        }
    }
}
