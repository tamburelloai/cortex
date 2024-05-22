//
// Created by Michael Tamburello on 5/19/24.
//

#ifndef CORTEX_PARAMETER_H
#define CORTEX_PARAMETER_H
#include "matrix.h"
#include "init_types.h"

namespace NN {
    class Parameter {
    public:
        Matrix data;
        Matrix grad;
        Parameter(size_t m, size_t n, InitType initType = InitType::Random);
        void initialize(InitType initType);

        // Accessors
        Matrix& getData();
        const Matrix& getData() const;
        Matrix& getGrad();
        const Matrix& getGrad() const;

        // Methods (updating and gradient management)
        void update(double alpha);
        void zeroGradients();
        void accumulateGradients(const Matrix& delta);
        void initializeRandom();
        void initializeZeros();
    };
}



#endif //CORTEX_PARAMETER_H
