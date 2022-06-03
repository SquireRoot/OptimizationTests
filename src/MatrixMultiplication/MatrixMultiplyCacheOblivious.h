/*
MatrixMultiplyCacheOblivious.h
Evan Newman
*/

#ifndef MATRIX_MULTIPLY_CACHE_OBLIVIOUS_H
#define MATRIX_MULTIPLY_CACHE_OBLIVIOUS_H

#include <Core>

namespace OptimizationTests {
    namespace MatrixMultiply {

        /** Performs a*b = c with a cache oblivious recursive algorithm
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMultCacheOblivious(const Eigen::Ref<const Eigen::MatrixXf> a,
                                   const Eigen::Ref<const Eigen::MatrixXf> b,
                                   Eigen::Ref<Eigen::MatrixXf> c);
        
        /** Performs a*b = c with a cache oblivious recursive algorithm with coarse base case
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMultCacheObliviousOptimized(const Eigen::Ref<const Eigen::MatrixXf> a,
                                            const Eigen::Ref<const Eigen::MatrixXf> b,
                                            Eigen::Ref<Eigen::MatrixXf> c);
    
    } // namespace MatrixMultiply
} // namespace OptimizationTests

#endif // #ifndef MATRIX_MULTIPLY_CACHE_OBLIVIOUS_H
