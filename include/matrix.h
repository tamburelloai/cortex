//
// Created by Michael Tamburello on 5/19/24.
//

#ifndef NN_CPP_MATRIX_H
#define NN_CPP_MATRIX_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>

using namespace std;

class Matrix {
private:
    vector<vector<double>> data;
    size_t m, n;

public:
    Matrix() : m(0), n(0), data() {} // Default constructor
    Matrix(size_t m, size_t n);

    Matrix(initializer_list<initializer_list<double>> initData);

    Matrix(vector<vector<double>> initData);

    Matrix(vector<vector<int>> initData);

    Matrix(initializer_list<double> initData);

    Matrix(vector<double> initData);

    double &operator()(size_t i, size_t j);

    const double &operator()(size_t i, size_t j) const;

    double getElement(size_t i, size_t j) const;

    size_t numRows() const;

    size_t numCols() const;

    Matrix transpose() const;

    void print() const;

    Matrix apply(std::function<double(double)> func) const;


    Matrix operator+(const Matrix &other) const; // Overload the + operator for addition
    Matrix operator-(const Matrix &other) const; // Overload the - operator for subtraction
    Matrix operator*(const Matrix &other) const; // Overload the * operator for hadamard
    bool operator==(const Matrix &other) const; // Overload the * operator for hadamard

    friend Matrix operator*(const Matrix &mat, double scalar); // lhs and rhs scalar mult of matrix
    friend Matrix operator*(double scalar, const Matrix &mat); // lhs and rhs scalar mult of matrix

    Matrix matMul(const Matrix &other);

    Matrix matMul(const Matrix &other) const; // Note the const at the end of the declaration

    Matrix operator/(double scalar) const; // Element-wise scalar division
    Matrix &operator/=(double scalar); // In-place element-wise scalar division



    static Matrix identity(size_t n);

    void initializeRandom();

    void initializeZeros();

    void fill(double value);

    Matrix scale(double scalar) const;

    void subtractInPlace(const Matrix &other);

    double sum();


};


#endif //NN_CPP_MATRIX_H
