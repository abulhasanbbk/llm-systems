#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<float> data;

public:
    Matrix(int rows, int cols);

    void set(int row, int col, float value);

    float get(int row, int col) const;

    void print() const;

    Matrix multiply(const Matrix& other) const;

    Matrix add(const Matrix& other) const;
    
    Matrix transpose() const;
};

#endif