//
// Created by Michael Tamburello on 5/19/24.
//

#ifndef CORTEX_LINEAR_H
#define CORTEX_LINEAR_H
#include "../include/parameter.h"
#include "../include/layers/layer.h"
#include "../matrix.h"
#include <vector>
#include <iostream>

namespace NN {
    class Linear : public Layer {
    private:
        Parameter weights;
        Parameter bias;
        Matrix input;
        Matrix output;

    public:
        Linear(size_t inputSize, size_t outputSize, InitType initType);
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradOutput) override;


        Parameter& getWeights() { return weights; }
        Parameter& getBias() { return bias; }

        std::vector<Parameter*> params() override;
       //
    };

}


#endif //CORTEX_LINEAR_H
