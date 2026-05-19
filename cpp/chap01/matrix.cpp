#include <iostream>
#include "matrix.hpp"

Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f) {}

void Matrix::set(int row, int col, float value) {
    data[row * cols + col] = value;
}

float Matrix::get(int row, int col) const {
    return data[row * cols + col];
}

void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << get(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::multiply(const Matrix& other) const {

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < other.cols; j++) {

            float sum = 0.0f;

            for (int k = 0; k < cols; k++) {
                sum += get(i, k) * other.get(k, j);
            }

            result.set(i, j, sum);
        }
    }

    return result;
}