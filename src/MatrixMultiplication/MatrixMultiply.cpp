/*
MatrixMultiply.cpp
Evan Newman
*/

#include "MatrixMultiply.h"

#include <stdexcept>
#include <iostream>
#include <cstdint>

#include <Core> // Eigen stuff

#include "Util/Timer.h"

namespace OptimizationTests {
    namespace MatrixMultiply {

        void RunMatrixMultiplyTests() {
            std::cout << "-------- MatrixMultiply Tests --------\n";

            uint64_t dim1 = 100; // rows of c and a
            uint64_t dim2 = 100; // columns of c and b
            uint64_t dim3 = 100; // columns of a and rows of b

            const int num_iter = 50;

            Eigen::MatrixXf a(dim1, dim3);
            Eigen::MatrixXf b(dim3, dim2);
            Eigen::MatrixXf c(dim1, dim2);

            a.setRandom();
            b.setRandom();
            c.setZero();

            Util::Timer timer;

            // run and time eigen
            Eigen::MatrixXf c_eigen(dim1, dim2);

            timer.Start();
            for (int i = 0; i < num_iter; i++) {
                c_eigen = (a*b).eval();
            }
            double time_eigen = timer.Stop()/num_iter;
            std::cout << "Eigen:   \t" << time_eigen << "ms\n";

            auto RunTest = [&](auto func, std::string label) {
                c.setZero();

                timer.Start();
                for (int i = 0; i < num_iter; i++) {
                    func(a, b, c);
                }
                double dt = timer.Stop()/num_iter;
                std::cout << label << ":\t" << dt << "ms, result ";

                // test output
                if (c.isApprox(c_eigen)) {
                    std::cout << "ok\n";
                } else {
                    std::cout << "error!\n";
                }
            };

            /* ----- Test the functions ----- */
            RunTest(MatMult0, "MatMult0");
            RunTest(MatMult1, "MatMult1");
            RunTest(MatMult2, "MatMult2");
            RunTest(MatMult3, "MatMult3");
            RunTest(MatMult4, "MatMult4");
            RunTest(MatMult5, "MatMult5");
        }

        void MatMult0(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {
            
            // Ensure the inpus are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            /* naive approach */
            for (int c_row = 0; c_row < c.rows(); c_row++) { // for every row in the output c
                for (int c_col = 0; c_col < c.cols(); c_col++) { // for every column in the output c
                    c_raw[c_row + c_col*c.rows()] = 0; // initialize c to 0 before we sum

                    for (int i = 0; i < a.cols(); i++) { // for every column of a / row of b
                        // multiply the element of a and the element of b and add the result to the current value of c
                        c_raw[c_row + c_col*c.rows()] += a_raw[c_row + i*a.rows()]*b_raw[i + c_col*b.rows()];
                    }
                }
            }
        }

        void MatMult1(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {
            
            // Ensure the inpus are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            /* cache the value of the sum instead of summing and
             * indexing the c array directly for each element of
             * the i iteration
            */
            for (int c_row = 0; c_row < c.rows(); c_row++) { // for every row of c
                for (int c_col = 0; c_col < c.cols(); c_col++) { // for every column of c
                    float sum = 0; // initialize the running sum to 0

                    for (int i = 0; i < a.cols(); i++) { // for every column of a / row of b
                        // multiply the element of a and the element of b and add the result to the current value of sum
                        sum += a_raw[c_row + i*a.rows()]*b_raw[i + c_col*b.rows()];
                    }

                    // set the value of c to be the running sum
                    c_raw[c_row + c_col*c.rows()] = sum;
                }
            }
        }

        void MatMult2(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {

            // Ensure the inpus are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            /* Swap the first two loops so that c is indexed linearly */
            for (int c_col = 0; c_col < c.cols(); c_col++) { // for every column of c
                for (int c_row = 0; c_row < c.rows(); c_row++) { // for every row of c
                    float sum = 0; // initialize the running sum to 0

                    for (int i = 0; i < a.cols(); i++) { // for every column of a / row of b
                        // multiply the element of a and the element of b and add the result to the current value of sum
                        sum += a_raw[c_row + i*a.rows()]*b_raw[i + c_col*b.rows()];
                    }

                    // set the value of c to be the running sum
                    c_raw[c_row + c_col*c.rows()] = sum;
                }
            }
        }

        void MatMult3(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {

            const int block_size = 10;

            // Ensure the inpus are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            /* use submatrix tiling */
            for (int c_col = 0; c_col < c.cols(); c_col += block_size) { // for every column in c, incremented by block_size
                // if a block_size width block goes past the end of the columns,
                // set the block width equal to the remaining number of columns to the end
                // otherwise the block with is just the block size
                int block_width = c_col + block_size >= c.cols() ? c.cols() - c_col : block_size;
                
                for (int c_row = 0; c_row < c.rows(); c_row += block_size) {
                    // same scheme as for the columns and block_width
                    int block_height = c_row + block_size >= c.rows() ? c.rows() - c_row : block_size; 
                    
                    // initialize the current block of c to 0
                    for (int c_col_block = 0; c_col_block < block_width; c_col_block++)
                        for (int c_row_block = 0; c_row_block < block_height; c_row_block++)
                            c_raw[c_row + c_row_block + (c_col + c_col_block)*c.rows()] = 0;

                    for (int i = 0; i < a.cols(); i += block_size) { // for every column of a / row of c
                        // same scheme as for the columns and block_width
                        int block_i_size = i + block_size >= a.cols() ? a.cols() - i : block_size;

                        // multiply the a and b submatrices together and put the result in c
                        for (int c_col_block = 0; c_col_block < block_width; c_col_block++) { // for every column of the block
                            for (int c_row_block = 0; c_row_block < block_height; c_row_block++) { // for every row of the block
                                float sum = 0;
                                
                                for (int i_block = 0; i_block < block_i_size; i_block++) { // for every inner block dimension yadda yadda
                                    sum += a_raw[c_row + c_row_block + (i + i_block)*a.rows()]*b_raw[i + i_block + (c_col + c_col_block)*b.rows()];
                                }

                                c_raw[c_row + c_row_block + (c_col + c_col_block)*c.rows()] += sum;
                            }
                        }
                    }
                }
            }
        }

        void MatMult4(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {

            const int block_size = 10;

            // Ensure the inpus are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            /* use submatrix tiling with better indexing */
            for (int c_col = 0; c_col < c.cols(); c_col += block_size) {
                int block_width = c_col + block_size >= c.cols() ? c.cols() - c_col : block_size;
                
                for (int c_row = 0; c_row < c.rows(); c_row += block_size) {
                    int block_height = c_row + block_size >= c.rows() ? c.rows() - c_row : block_size; 
                    
                    for (int c_col_block = 0; c_col_block < block_width; c_col_block++)
                        for (int c_row_block = 0; c_row_block < block_height; c_row_block++)
                            c_raw[c_row + c_row_block + (c_col + c_col_block)*c.rows()] = 0;

                    for (int i = 0; i < a.cols(); i += block_size) {
                        int block_i_size = i + block_size >= a.cols() ? a.cols() - i : block_size;
 
                        for (int c_col_block = 0; c_col_block < block_width; c_col_block++) {
                            for (int c_row_block = 0; c_row_block < block_height; c_row_block++) {
                                float sum = 0;
                                
                                const float* a_raw_current = a_raw + c_row + c_row_block + i*a.rows();
                                const float* b_raw_current = b_raw + i + (c_col + c_col_block)*b.rows();
                                for (int i_block = 0; i_block < block_i_size; i_block++, a_raw_current += a.rows(), b_raw_current++) {
                                    sum += (*a_raw_current)*(*b_raw_current);
                                }

                                c_raw[c_row + c_row_block + (c_col + c_col_block)*c.rows()] += sum;
                            }
                        }
                    }
                }
            }
        }

        void MatMult5(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {

            const int block_size = 50; // my machine performed best with this 

            // Ensure the inpus are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            /* use submatrix tiling with better indexing */
            for (int c_col = 0; c_col < c.cols(); c_col += block_size) {
                int block_width = c_col + block_size >= c.cols() ? c.cols() - c_col : block_size;
                
                for (int c_row = 0; c_row < c.rows(); c_row += block_size) {
                    int block_height = c_row + block_size >= c.rows() ? c.rows() - c_row : block_size; 
                    
                    for (int c_col_block = 0; c_col_block < block_width; c_col_block++)
                        for (int c_row_block = 0; c_row_block < block_height; c_row_block++)
                            c_raw[c_row + c_row_block + (c_col + c_col_block)*c.rows()] = 0;

                    for (int i = 0; i < a.cols(); i += block_size) {
                        int block_i_size = i + block_size >= a.cols() ? a.cols() - i : block_size;
 
                        for (int c_col_block = 0; c_col_block < block_width; c_col_block++) {
                            for (int c_row_block = 0; c_row_block < block_height; c_row_block++) {
                                float sum = 0;
                                
                                const float* a_raw_current = a_raw + c_row + c_row_block + i*a.rows();
                                const float* b_raw_current = b_raw + i + (c_col + c_col_block)*b.rows();
                                for (int i_block = 0; i_block < block_i_size; i_block++, a_raw_current += a.rows(), b_raw_current++) {
                                    sum += (*a_raw_current)*(*b_raw_current);
                                }

                                c_raw[c_row + c_row_block + (c_col + c_col_block)*c.rows()] += sum;
                            }
                        }
                    }
                }
            }
        }
    } // namespace MatrixMultiply

} // namepsace OptimizationTests