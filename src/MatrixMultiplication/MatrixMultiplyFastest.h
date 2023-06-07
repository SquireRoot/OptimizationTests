/*
MatrixMultiplyTiled.h
Evan Newman
*/

#ifndef MATRIX_MULTIPLY_TILED_H
#define MATRIX_MULTIPLY_TILED_H

#include <eigen3/Eigen/Core>

namespace OptimizationTests {
    namespace MatrixMultiply {

        /** Performs a*b = c with submatrix tiling
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMultFastest(const Eigen::Ref<const Eigen::MatrixXf> a,
                            const Eigen::Ref<const Eigen::MatrixXf> b,
                            Eigen::Ref<Eigen::MatrixXf> c);

    } // namespace MatrixMultiply
} // namespace OptimizationTests

#endif // #ifndef MATRIX_MULTIPLY_TILED_H
