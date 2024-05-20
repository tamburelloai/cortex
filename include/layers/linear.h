//
// Created by Michael Tamburello on 5/19/24.
//

#ifndef CORTEX_LINEAR_H
#define CORTEX_LINEAR_H
#include "../include/parameter.h"
#include "../matrix.h"

namespace NN {
    class Linear {
    private:
        Parameter weights;
        Parameter bias;
        Matrix input;
        Matrix output;

    public:
        Linear(size_t inputSize, size_t outputSize, InitType initType);
        Matrix forward(const Matrix& input);
        void backward();

        Parameter& getWeights() { return weights; }
        Parameter& getBias() { return bias; }
        //
    };

}


#endif //CORTEX_LINEAR_H
