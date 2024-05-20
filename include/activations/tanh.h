//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_TANH_H
#define CORTEX_TANH_H

#include "activation.h"

class Tanh : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
};

#endif //CORTEX_TANH_H
