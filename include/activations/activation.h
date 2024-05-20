//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_ACTIVATION_H
#define CORTEX_ACTIVATION_H

#include "matrix.h"

class ActivationFunction {
public:
    virtual ~ActivationFunction() {}
    virtual Matrix forward(const Matrix& input) const = 0;
    virtual Matrix backward(const Matrix& input) const = 0;
};

#endif //CORTEX_ACTIVATION_H

