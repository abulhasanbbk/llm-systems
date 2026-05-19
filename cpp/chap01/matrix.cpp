#include <iostream>
#include "matrix.hpp"
#include <stdexcept>
#include <random>
Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f) {}

void Matrix::set(int row, int col, float value) {
    data[row * cols + col] = value;
}

float Matrix::get(int row, int col) const {
    return data[row * cols + col];
}

void Matrix::randomize(float min, float max) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (int i = 0; i < rows * cols; i++) {
        data[i] = dist(gen);
    }
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
    
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns of A must match number of rows of B.");
    }

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

Matrix Matrix::add(const Matrix& other) const {
    
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float value = get(i, j) + other.get(i, j);
            result.set(i, j, value);
        }
    }

    return result;
}

Matrix Matrix::transpose() const {

    Matrix result(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.set(j, i, get(i, j));
        }
    }

    return result;
}