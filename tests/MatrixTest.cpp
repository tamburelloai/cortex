//
// Created by Michael Tamburello on 5/19/24.
//

#include "../include/matrix.h"
#include <gtest/gtest.h>

// Test for Matrix Multiplication
TEST(MatrixTest, MatrixMultiplication) {
Matrix A = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix B = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix expected = {{30, 36, 42},
                   {66, 81, 96},
                   {102, 126, 150}};
Matrix result = A.matMul(B);
ASSERT_TRUE(result == expected);
}

TEST(MatrixTest, FillMethod) {
    double value = 4.0;
    Matrix A = {{1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}};
    A.fill(value);
    for (size_t i = 0; i < A.numRows(); i++) {
        for (size_t j = 0; j < A.numCols(); j++) {
            ASSERT_EQ(A.getElement(i, j), value);
        }
    }
}

// Test for Matrix Multiplication with non-square matrices
TEST(MatrixTest, MatrixMultiplicationNonSquare) {
Matrix A = {{1, 2, 3}};
Matrix B = {{1, 2},
            {3, 4},
            {5, 6}};
Matrix expected = {{22, 28}};
Matrix result = A.matMul(B);
ASSERT_TRUE(result == expected);
}

// Test for Scalar Multiplication
TEST(MatrixTest, ScalarMultiplication) {
Matrix A = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
double scalar = 2.0;
Matrix expected = {{2, 4, 6},
                   {8, 10, 12},
                   {14, 16, 18}};
Matrix result = A * scalar;
ASSERT_TRUE(result == expected);
}

// Test for Element-wise Multiplication (Hadamard Product)
TEST(MatrixTest, HadamardProduct) {
Matrix A = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix B = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix expected = {{1, 4, 9},
                   {16, 25, 36},
                   {49, 64, 81}};
Matrix result = A * B;
ASSERT_TRUE(result == expected);
}

// Test for Matrix Addition
TEST(MatrixTest, MatrixAddition) {
Matrix A = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix B = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix expected = {{2, 4, 6},
                   {8, 10, 12},
                   {14, 16, 18}};
Matrix result = A + B;
ASSERT_TRUE(result == expected);
}

// Test for Matrix Subtraction
TEST(MatrixTest, MatrixSubtraction) {
Matrix A = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix B = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix expected = {{0, 0, 0},
                   {0, 0, 0},
                   {0, 0, 0}};
Matrix result = A - B;
ASSERT_TRUE(result == expected);
}

// Test for Identity Matrix Multiplication
TEST(MatrixTest, IdentityMatrixMultiplication) {
Matrix A = {{1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}};
Matrix identity = Matrix::identity(3);
Matrix result = A.matMul(identity);
ASSERT_TRUE(result == A);
}
