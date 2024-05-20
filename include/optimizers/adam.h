//
// Created by Michael Tamburello on 5/20/24.
//

#ifndef CORTEX_ADAM_H
#define CORTEX_ADAM_H
#include <vector>
#include "optimizer.h"
#include "matrix.h"

class Adam : public Optimizer {
private:
    double learningRate, beta1, beta2, epsilon;
    std::vector<Matrix> m, v;
    int t;
public:
    Adam(double lr, double b1, double b2, double eps);
    void update() override;


};


#endif //CORTEX_ADAM_H
