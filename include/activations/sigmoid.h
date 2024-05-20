//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_SIGMOID_H
#define CORTEX_SIGMOID_H

#include "activation.h"
#include "matrix.h"
#include <cmath>

class Sigmoid : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
};

#endif //CORTEX_SIGMOID_H
