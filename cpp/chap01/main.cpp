#include <iostream>
#include "matrix.hpp"
#include <chrono>

int main() {

    Matrix A(2, 2);
    Matrix B(2, 2);

    A.set(0, 0, 1);
    A.set(0, 1, 2);
    A.set(1, 0, 3);
    A.set(1, 1, 4);

    B.set(0, 0, 5);
    B.set(0, 1, 6);
    B.set(1, 0, 7);
    B.set(1, 1, 8);

    std::cout << "A x B:" << std::endl;
    Matrix C = A.multiply(B);
    C.print();

    std::cout << "A + B:" << std::endl;
    Matrix D = A.add(B);
    D.print();

    std::cout << "Transpose of A:" << std::endl;
    Matrix At = A.transpose();
    At.print();

    Matrix W(3, 3);
    W.randomize(-1.0f, 1.0f);

    std::cout << "Random matrix W:" << std::endl;
    W.print();

    Matrix X(300, 300);
    Matrix Y(300, 300);

    X.randomize(-1.0f, 1.0f);
    Y.randomize(-1.0f, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();

    Matrix Z = X.multiply(Y);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time for 300x300 matrix multiplication: "
          << elapsed.count()
          << " seconds"
          << std::endl;
    return 0;
}