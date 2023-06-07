/*
MatrixMultiplyTiled.cpp
Evan Newman
*/

#include <eigen3/Eigen/Core>

namespace OptimizationTests {
    namespace MatrixMultiply {

        void MatMultTiled(const Eigen::Ref<const Eigen::MatrixXf> a,
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

            for (uint64_t i = 0; i < c.rows()*c.cols(); i++) c_raw[i] = 0;

            /* use submatrix tiling */
            for (int c_col = 0; c_col < c.cols(); c_col += block_size) { // for every column in c, incremented by block_size
                // if a block_size width block goes past the end of the columns,
                // set the block width equal to the remaining number of columns to the end
                // otherwise the block with is just the block size
                int block_width = c_col + block_size >= c.cols() ? c.cols() - c_col : block_size;
                
                for (int c_row = 0; c_row < c.rows(); c_row += block_size) {
                    // same scheme as for the columns and block_width
                    int block_height = c_row + block_size >= c.rows() ? c.rows() - c_row : block_size; 
                    
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


        void MatMultTiledOptimized(const Eigen::Ref<const Eigen::MatrixXf> a,
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

            for (uint64_t i = 0; i < c.rows()*c.cols(); i++) c_raw[i] = 0;

            /* use submatrix tiling with better indexing */
            for (int c_col = 0; c_col < c.cols(); c_col += block_size) {
                int block_width = c_col + block_size >= c.cols() ? c.cols() - c_col : block_size;
                
                for (int c_row = 0; c_row < c.rows(); c_row += block_size) {
                    int block_height = c_row + block_size >= c.rows() ? c.rows() - c_row : block_size; 

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
} // namespace OptimizationTests