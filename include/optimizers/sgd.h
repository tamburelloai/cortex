//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_SGD_H
#define CORTEX_SGD_H
#include "optimizer.h"


class SGD : public Optimizer {
private:
    double learningRate;
public:
    SGD(double lr);
    void update() override;
};


#endif //CORTEX_SGD_H
