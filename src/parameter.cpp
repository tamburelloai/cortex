//
// Created by Michael Tamburello on 5/19/24.
//

#include "../include/parameter.h"
#include <random>

namespace NN {
    Parameter::Parameter(size_t m, size_t n, InitType initType)
        : data(m, n), grad(m, n) {
        initialize(initType);
    }

    void Parameter::initialize(InitType initType) {
        std::random_device rd;
        std::mt19937 gen(rd());

        switch (initType) {
            case InitType::Zeros:
                data.initializeZeros();
                break;
            case InitType::Random:
                data.initializeRandom();  // Assuming this fills with uniform random values
                break;
            case InitType::Xavier: {
                double stddevXavier = std::sqrt(6.0 / (data.numRows() + data.numCols()));
                std::normal_distribution<> distXavier(0, stddevXavier);
                data.apply([&](double) { return distXavier(gen); });
                break;
            }
            case InitType::He: {
                double stddevHe = std::sqrt(2.0 / data.numRows());
                std::normal_distribution<> distHe(0, stddevHe);
                data.apply([&](double) { return distHe(gen); });
                break;
            }
        }
        grad.initializeZeros();
    }

    // Accessor Methods
    Matrix& Parameter::getData() {
        return data;
    }

    const Matrix& Parameter::getData() const {
        return data;
    }

    Matrix& Parameter::getGrad() {
        return grad;
    }

    const Matrix& Parameter::getGrad() const {
        return grad;
    }

    // Update / Management of Gradient Methods
    void Parameter::zeroGradients() {
        grad.initializeZeros();
    }

    void Parameter::accumulateGradients(const Matrix &delta) {
        grad = grad + delta;
    }

    void Parameter::update(double alpha) {
        data = data - (alpha*grad);
    }

    void Parameter::initializeRandom() {
        data.initializeRandom();
    }
} // namespace NN

