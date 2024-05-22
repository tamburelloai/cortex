//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_LOSS_H
#define CORTEX_LOSS_H

#include "matrix.h"  // Assuming Matrix is your data structure for storing tensors

namespace NN {

    class Loss {
    public:
        virtual ~Loss() {}

        // Compute the loss value
        virtual double compute(const Matrix& predicted, const Matrix& actual) = 0;

        // Compute the gradient of the loss with respect to the outputs of the model
        virtual Matrix gradient(const Matrix& predicted, const Matrix& actual) = 0;
    };

}

#endif //CORTEX_LOSS_H
