//
// Created by Michael Tamburello on 5/19/24.
//

#include <gtest/gtest.h>
#include "../include/parameter.h"

// Test the constructor and initialization
TEST(ParameterTest, ConstructorInitializationGradIsZero) {
    NN::Parameter param(3, 2);  // 3x2 matrix
    ASSERT_EQ(param.getData().numRows(), 3);
    ASSERT_EQ(param.getData().numCols(), 2);
    // Assume initializeZeros has been called in the constructor
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            ASSERT_EQ(param.getGrad()(i, j), 0);
        }
    }
}

// Test random initialization
TEST(ParameterTest, RandomInitialization) {
    NN::Parameter param(3, 2);
    param.getData().initializeRandom(); // Manually initialize to random for test
    bool allZero = true;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (param.getData()(i, j) != 0) {
                allZero = false;
                break;
            }
        }
        if (!allZero) break;
    }
    ASSERT_FALSE(allZero);
}

// Test updating Parameters
TEST(ParameterTest, UpdateParameters) {
    NN::Parameter param(2, 2);
    // Set initial values and gradients
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            param.getData()(i, j) = 1.0;
            param.getGrad()(i, j) = 0.1;
        }
    }
    // Perform update with learning rate 0.5
    param.update(0.5);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            ASSERT_NEAR(param.getData()(i, j), 0.95, 0.001);
        }
    }
}
