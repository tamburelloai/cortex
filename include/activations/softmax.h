//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_SOFTMAX_H
#define CORTEX_SOFTMAX_H
#include "activation.h"

class Softmax : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
};

#endif //CORTEX_SOFTMAX_H
