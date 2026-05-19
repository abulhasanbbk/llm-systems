#include <iostream>
#include "matrix.hpp"

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

    return 0;
}