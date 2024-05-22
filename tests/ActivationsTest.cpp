// TODO: Implement ReLU tests
// TODO: Implement Sigmoid tests
// TODO: Implement Tanh tests
// TODO: Implement Softmax tests

#include "../include/activations/relu.h"
#include "../include/activations/softmax.h"
#include <gtest/gtest.h>


TEST(ReLUForward, HandlesPositiveValues) {
    Matrix input{{1.0, 2.0},
                 {3.0, 4.0}};
    NN::ReLU relu;
    auto output = relu.forward(input);
    EXPECT_DOUBLE_EQ(output(0,0), 1.0);
    EXPECT_DOUBLE_EQ(output(0,1), 2.0);
    EXPECT_DOUBLE_EQ(output(1,0), 3.0);
    EXPECT_DOUBLE_EQ(output(1,1), 4.0);
}

TEST(ReLUForward, HandlesNegativeValues) {
    Matrix input{{-1.0, -2.0},
                 {-3.0, -4.0}};
    NN::ReLU relu;
    auto output = relu.forward(input);
    EXPECT_DOUBLE_EQ(output(0,0), 0.0);
    EXPECT_DOUBLE_EQ(output(0,1), 0.0);
    EXPECT_DOUBLE_EQ(output(1,0), 0.0);
    EXPECT_DOUBLE_EQ(output(1,1), 0.0);
}

TEST(ReLUForward, HandlesMixedValues) {
    Matrix input{{-1.0, 2.0},
                 {-3.0, 4.0}};
    NN::ReLU relu;
    auto output = relu.forward(input);
    EXPECT_DOUBLE_EQ(output(0,0), 0.0);
    EXPECT_DOUBLE_EQ(output(0,1), 2.0);
    EXPECT_DOUBLE_EQ(output(1,0), 0.0);
    EXPECT_DOUBLE_EQ(output(1,1), 4.0);
}

TEST(ReLUBackward, PropagateGradients) {
    Matrix input{{-1.0, 2.0},
                 {-3.0, 4.0}};
    Matrix gradOutput{{0.5, 0.5},
                      {0.5, 0.5}}; // Example gradient matrix from subsequent layers
    NN::ReLU relu;
    relu.forward(input); // Set internal state
    auto gradientInput = relu.backward(gradOutput);

    EXPECT_DOUBLE_EQ(gradientInput(0,0), 0.0); // Gradient should not pass for negative inputs
    EXPECT_DOUBLE_EQ(gradientInput(0,1), 0.5); // Gradient should pass for positive inputs
    EXPECT_DOUBLE_EQ(gradientInput(1,0), 0.0); // Gradient should not pass for negative inputs
    EXPECT_DOUBLE_EQ(gradientInput(1,1), 0.5); // Gradient should pass for positive inputs
}



class SoftmaxTest : public ::testing::Test {
protected:
    NN::Softmax softmax;
};

TEST_F(SoftmaxTest, OutputsValidProbabilities) {
    Matrix input{{1.0, 2.0, 3.0}};
    auto output = softmax.forward(input);
    double sum = 0.0;
    for (int j = 0; j < output.cols(); ++j) {
        sum += output(0, j);
        ASSERT_GE(output(0, j), 0.0);  // Each output probability should be non-negative
        ASSERT_LE(output(0, j), 1.0);  // and less than or equal to 1
    }
    ASSERT_NEAR(sum, 1.0, 1e-6);  // The sum of probabilities should be 1
}

TEST_F(SoftmaxTest, HandlesLargeValues) {
    Matrix input{{1000.0, 1000.0, 1000.0}};
    auto output = softmax.forward(input);
    double sum = 0.0;
    for (int j = 0; j < output.cols(); ++j) {
        sum += output(0, j);
    }
    ASSERT_NEAR(sum, 1.0, 1e-6);  // The sum of probabilities should still be 1
}

TEST_F(SoftmaxTest, HandlesUniformInputs) {
    Matrix input{{2.0, 2.0, 2.0}};
    auto output = softmax.forward(input);
    for (int j = 0; j < output.cols(); ++j) {
        ASSERT_NEAR(output(0, j), 1.0 / 3.0, 1e-6);  // Each probability should be equal
    }
}

TEST_F(SoftmaxTest, HandlesNegativeValues) {
    Matrix input{{-1.0, -2.0, -3.0}};
    auto output = softmax.forward(input);
    double sum = 0.0;
    for (int j = 0; j < output.cols(); ++j) {
        sum += output(0, j);
        ASSERT_GE(output(0, j), 0.0);  // Each output probability should be non-negative
    }
    ASSERT_NEAR(sum, 1.0, 1e-6);  // The sum of probabilities should be 1
}


class SoftmaxBackwardTest : public ::testing::Test {
protected:
    NN::Softmax softmax;
};

TEST_F(SoftmaxBackwardTest, CorrectGradientComputation) {
    Matrix input{{1.0, 2.0, 3.0}};
    Matrix target{{0.0, 0.0, 1.0}}; // Assuming the correct class is the third one

    // Forward to set the state
    auto output = softmax.forward(input);

    // Compute backward pass
    auto gradInput = softmax.backward(target);

    // Manually compute expected gradients
    // output is the softmax result, and target is the one-hot encoded true labels
    // Gradient should be output - target
    Matrix expectedGrad = output - target;

    for (int i = 0; i < gradInput.rows(); ++i) {
        for (int j = 0; j < gradInput.cols(); ++j) {
            ASSERT_NEAR(gradInput(i, j), expectedGrad(i, j), 1e-6);
        }
    }
}

TEST_F(SoftmaxBackwardTest, GradientSum) {
    Matrix input{{1.0, 1.0, 1.0}};
    Matrix target{{0.0, 1.0, 0.0}}; // Assuming the correct class is the second one

    // Forward to set the state
    auto output = softmax.forward(input);

    // Compute backward pass
    auto gradInput = softmax.backward(target);

    // For each example, the sum of gradients should be zero if softmax output directly follows cross-entropy
    double sum = 0.0;
    for (int j = 0; j < gradInput.cols(); ++j) {
        sum += gradInput(0, j);
    }
    ASSERT_NEAR(sum, 0.0, 1e-6);
}