//
// Created by Michael Tamburello on 5/21/24.
//

#ifndef CORTEX_MSE_LOSS_H
#define CORTEX_MSE_LOSS_H

#include "loss.h"

namespace NN {

    class MSELoss : public Loss {
    public:
        double compute(const Matrix& predicted, const Matrix& actual) override {
            Matrix diff = predicted - actual;
            Matrix sdiff = diff * diff;
            return sdiff.sum();
        }

        Matrix gradient(const Matrix& predicted, const Matrix& actual) override {
            Matrix gradient;
            gradient = predicted - actual;
            gradient = 2 * gradient;
            gradient = gradient * (1/predicted.numRows());
            return gradient;
        }
    };

}



#endif //CORTEX_MSE_LOSS_H
