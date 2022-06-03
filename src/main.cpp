/*
main.cpp its the main
Evan Newman
*/

#include <iostream>

#include <Core> // Eigen stuff

#include "MatrixMultiplication/MatrixMultiply.h"

using namespace OptimizationTests;

int main(int argc, char** argv) {

    MatrixMultiply::RunMatrixMultiplyTests();

    // uint64_t dim = 6;

    // Eigen::MatrixXf a(dim, dim);
    // Eigen::MatrixXf b(dim, dim);
    // Eigen::MatrixXf c(dim, dim);

    // for (int i = 0; i < dim*dim; i++) {
    //     a.data()[i] = i;
    //     b.data()[i] = i;
    //     c.data()[i] = i;
    // }

    // // a.setRandom();
    // // b.setRandom();
    // // c.setZero();

    // MatrixMultiply::MatMult6(a, b, c);

    return 0;
}