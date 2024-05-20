//
// Created by Michael Tamburello on 5/20/24.
//

#include "optimizers/sgd.h"


SGD::SGD(double lr) : learningRate(lr) {}

void SGD::update() {
    for (auto param : parameters) {
        param->getData().subtractInPlace(param->getGrad().scale(learningRate));
    }
}