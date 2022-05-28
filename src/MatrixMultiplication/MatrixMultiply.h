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
        void MatMult0(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);
        
        /** Performs a*b = c with a cached sum
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult1(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);

        /** Performs a*b = c with linear c indexing
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult2(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);

        /** Performs a*b = c with submatrix tiling
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult3(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);

        /** Performs a*b = c with submatrix tiling and better 
         *  indexing in the inner loop
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult4(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);

        /** Performs a*b = c with submatrix tiling and better 
         *  indexing in the inner loop. A manual search
         *  was performed to optimize the block size
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult5(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);

        /** Performs a*b = c with submatrix tiling and better 
         *  indexing in the inner loop. A manual search
         *  was performed to optimize the block size
         * 
         * \param a the input matrix a
         * \param b the input matrix b
         * 
         * \return the resulting matrix c
         */
        void MatMult5(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c);
        

    } // namespace MatrixMultiply

} // namepsace OptimizationTests

#endif // MATRIX_MULTIPLY_H