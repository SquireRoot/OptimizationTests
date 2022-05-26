/*
MatrixMultiply.h is the header for the matrix multiplication test
Evan Newman
*/

#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <Core> // Eigen stuff

namespace OptimizationTests {
    namespace MatrixMultiply {

        /** runs all the matrix multiplication tests in an organized way
         */
        void RunMatrixMultiplyTests();

        /** Performs a*b = c without any optimizations
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult1(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);


    } // namespace MatrixMultiply

} // namepsace OptimizationTests

#endif // MATRIX_MULTIPLY_H