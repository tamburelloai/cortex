//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_RELU_H
#define CORTEX_RELU_H
#include "activation.h"

class ReLU : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input) const override;
};



#endif //CORTEX_RELU_H
