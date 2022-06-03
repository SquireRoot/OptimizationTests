/*
MatrixMultiplySimple.h
Evan Newman
*/

#ifndef MATRIX_MULTIPLY_SIMPLE_H
#define MATRIX_MULTIPLY_SIMPLE_H

#include <Core>

namespace OptimizationTests {
    namespace MatrixMultiply {
        
        /** Performs a*b = c without any optimizations
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMultSimple(const Eigen::Ref<const Eigen::MatrixXf> a,
                           const Eigen::Ref<const Eigen::MatrixXf> b,
                           Eigen::Ref<Eigen::MatrixXf> c);
        

        /** Performs a*b = c with linear c indexing and a cached sum
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMultSimpleOptimized(const Eigen::Ref<const Eigen::MatrixXf> a,
                                    const Eigen::Ref<const Eigen::MatrixXf> b,
                                    Eigen::Ref<Eigen::MatrixXf> c);

    } // namespace MatrixMultiply
} // namespace OptimizationTests

#endif // #ifndef MATRIX_MULTIPLY_SIMPLE_H