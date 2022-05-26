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

    return 0;
}