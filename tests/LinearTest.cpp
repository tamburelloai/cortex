
#include "../include/layers/linear.h"
#include "../include/init_types.h"
#include <gtest/gtest.h>


// Test the Linear constructor to ensure correct initialization
TEST(LinearTest, ConstructorInitializesWeightsAndBiases) {
    NN::Linear layer(10, 5, InitType::Random); // Assuming random initializes with non-zero values
    auto weights = layer.getWeights().getData();
    auto bias = layer.getBias().getData();

    // Check weights and biases are not empty and have expected dimensions
    ASSERT_EQ(weights.numRows(), 10);
    ASSERT_EQ(weights.numCols(), 5);
    ASSERT_EQ(bias.numRows(), 1);
    ASSERT_EQ(bias.numCols(), 5);

    // Optionally, check that values are initialized to zero if using InitType::Zeros
}



// Test the forward pass
TEST(LinearTest, ForwardPassCorrectOutput) {
    NN::Linear layer(3, 2, InitType::Zeros);
    Matrix input(1, 3); // 1x3 input vector
    input.fill(1.0); // Fill input with 1.0 for simplicity

    auto output = layer.forward(input);

    // Expect zero output from zero-initialized weights and biases
    Matrix expectedOutput(1, 2);
    expectedOutput.fill(0.0);
    ASSERT_TRUE(output == expectedOutput); // Ensure operator== is implemented for Matrix
}

