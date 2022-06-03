/*
MatrixMultiplyCacheOblivious.cpp
Evan Newman
*/

#include <Core>

namespace OptimizationTests {
    namespace MatrixMultiply {

        void MatMultCacheOblivious(const Eigen::Ref<const Eigen::MatrixXf> a,
                                   const Eigen::Ref<const Eigen::MatrixXf> b,
                                   Eigen::Ref<Eigen::MatrixXf> c) {
                          
            // Ensure the inputs are ok for matrix multiplication
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

            std::function<void(const float* a_raw_current, const float* b_raw_current, float* c_raw_current,
                               const uint64_t& row_size, const uint64_t& col_size, const uint64_t& k_size)> MatMultRecursive
                    = [&](const float* a_raw_current, const float* b_raw_current, float* c_raw_current,
                                                     const uint64_t& row_size, const uint64_t& col_size, const uint64_t& k_size) {
                    
                /* base cases */
                // if any dimensions are 0, the partition is invalid
                if (row_size == 0 || col_size == 0 || k_size == 0) {
                    return;

                } 
                
                // if all the dimensions are 1, multiply a and b at the current location
                else if (row_size == 1 && col_size == 1 && k_size == 1) {
                    (*c_raw_current) += (*a_raw_current)*(*b_raw_current);
                    return;
                }

                /* the following formula holds when a, b, and c are scalars or matrices
                 * we can exploit this with the recursive algorithm
                 *
                 * / c_11   c_12 \   / a_11   a_12 \   / b_11   b_12 \ 
                 * |             | = |             | * |             |
                 * \ c_21   c_22 /   \ a_21   a_22 /   \ b_21   b_22 /
                 *
                 *   / a_11*b_11   a_11*b_12 \   / a_12*b_21   a_12*b_22 \
                 * = |                       | + |                       |
                 *   \ a_21*b_11   a_21*b_12 /   \ a_22*b_21   a_22*b_22 /
                */

                /* partition each dimension into two sub-sizes
                 * place the odd amount on the p1 side, ie closer to 0
                 * p1 should always be >= p2
                */
                // row partition
                uint64_t row_size_p2 = row_size/2;
                uint64_t row_size_p1 = row_size_p2 + row_size%2;
               
                // column partition
                uint64_t col_size_p2 = col_size/2;
                uint64_t col_size_p1 = col_size_p2 + col_size%2;

                // k partition
                uint64_t k_size_p2 = k_size/2;
                uint64_t k_size_p1 = k_size_p2 + k_size%2;

                /* calculate the offsets requred for each submatrix */
                // k offsets
                uint64_t a_k_offset = k_size_p1*a.rows();
                uint64_t b_k_offset = k_size_p1;

                // row offsets
                uint64_t a_row_offset = row_size_p1;
                uint64_t c_row_offset = row_size_p1;

                // col offsets
                uint64_t c_col_offset = col_size_p1*c.rows();
                uint64_t b_col_offset = col_size_p1*b.rows();

                /* calculate the terms */
                // c_11 += a_11*b_11
                MatMultRecursive(a_raw_current,
                                 b_raw_current,
                                 c_raw_current,
                                 row_size_p1, col_size_p1, k_size_p1);

                // c_11 += a_12*b_21
                MatMultRecursive(a_raw_current + a_k_offset,
                                 b_raw_current + b_k_offset,
                                 c_raw_current,
                                 row_size_p1, col_size_p1, k_size_p2);

                // c_21 += a_21*b_11
                MatMultRecursive(a_raw_current + a_row_offset,
                                 b_raw_current,
                                 c_raw_current + c_row_offset,
                                 row_size_p2, col_size_p1, k_size_p1);

                // c_21 += a_22*b_21
                MatMultRecursive(a_raw_current + a_row_offset + a_k_offset,
                                 b_raw_current + b_k_offset,
                                 c_raw_current + c_row_offset, 
                                 row_size_p2, col_size_p1, k_size_p2);

                // c_12 += a_11*b_12
                MatMultRecursive(a_raw_current,
                                 b_raw_current + b_col_offset,
                                 c_raw_current + c_col_offset,
                                 row_size_p1, col_size_p2, k_size_p1);

                // c_12 += a_12*b_22
                MatMultRecursive(a_raw_current + a_k_offset,
                                 b_raw_current + b_col_offset + b_k_offset, 
                                 c_raw_current + c_col_offset,
                                 row_size_p1, col_size_p2, k_size_p2);

                // c_22 += a_21*b_12
                MatMultRecursive(a_raw_current + a_row_offset,
                                 b_raw_current + b_col_offset,
                                 c_raw_current + c_row_offset + c_col_offset,
                                 row_size_p2, col_size_p2, k_size_p1);

                // c_22 += a_22*b_22
                MatMultRecursive(a_raw_current + a_row_offset + a_k_offset,
                                 b_raw_current + b_col_offset + b_k_offset, 
                                 c_raw_current + c_row_offset + c_col_offset,
                                 row_size_p2, col_size_p2, k_size_p2);

            };

            // kickstart that recursion baby
            MatMultRecursive(a_raw, b_raw, c_raw,
                             a.rows(), b.cols(), a.cols());
        }

        void MatMultCacheObliviousOptimized(const Eigen::Ref<const Eigen::MatrixXf> a,
                                            const Eigen::Ref<const Eigen::MatrixXf> b,
                                            Eigen::Ref<Eigen::MatrixXf> c) {

            const uint64_t block_size = 10;
                          
            // Ensure the inputs are ok for matrix multiplication
            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            } 

            if (a.rows() < block_size || a.cols() < block_size || b.cols() < block_size) {
                throw std::invalid_argument("matrices a, b, and c must have dimensions larger than the block size " + std::to_string(block_size));
            }

            // grab the data pointers from eigen
            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            for (uint64_t i = 0; i < c.rows()*c.cols(); i++) c_raw[i] = 0;

            auto MatMult = [&a, &b, &c](const float* a_raw_submatrix, const float* b_raw_submatrix, float* c_raw_submatrix,
                                        const uint64_t& row_size, const uint64_t& col_size, const uint64_t& k_size){
                
                for (int c_col = 0; c_col < col_size; c_col++) { // for every column of c
                    for (int c_row = 0; c_row < row_size; c_row++) { // for every row of c
                        float sum = 0; // initialize the running sum to 0

                        for (int k = 0; k < k_size; k++) { // for every column of a / row of b
                            // multiply the element of a and the element of b and add the result to the current value of sum
                            sum += a_raw_submatrix[c_row + k*a.rows()]*b_raw_submatrix[k + c_col*b.rows()];
                        }

                        // set the value of c to be the running sum
                        c_raw_submatrix[c_row + c_col*c.rows()] += sum;
                    }
                }
            };

            std::function<void(const float* a_raw_current, const float* b_raw_current, float* c_raw_current,
                               const uint64_t& row_size, const uint64_t& col_size, const uint64_t& k_size)> MatMultRecursive
                    = [&](const float* a_raw_current, const float* b_raw_current, float* c_raw_current, 
                          const uint64_t& row_size, const uint64_t& col_size, const uint64_t& k_size) {
                    
                /* base cases */
                // if any dimensions are 0, the partition is invalid
                if (row_size == 0 || col_size == 0 || k_size == 0) {
                    return;

                } 
                
                // if all the dimensions are less than the block size, multiply submatrix a and b at the current location
                else if (row_size <= block_size && col_size <= block_size && k_size <= block_size) {
                    MatMult(a_raw_current, b_raw_current, c_raw_current, row_size, col_size, k_size);
                    return;
                }

                /* the following formula holds when a, b, and c are scalars or matrices
                 * we can exploit this with the recursive algorithm
                 *
                 * / c_11   c_12 \   / a_11   a_12 \   / b_11   b_12 \ 
                 * |             | = |             | * |             |
                 * \ c_21   c_22 /   \ a_21   a_22 /   \ b_21   b_22 /
                 *
                 *   / a_11*b_11   a_11*b_12 \   / a_12*b_21   a_12*b_22 \
                 * = |                       | + |                       |
                 *   \ a_21*b_11   a_21*b_12 /   \ a_22*b_21   a_22*b_22 /
                */

                /* partition each dimension into two sub-sizes
                 * place the odd amount on the p1 side, ie closer to 0
                 * p1 should always be >= p2
                */
                // row partition
                uint64_t row_size_p2 = row_size/2;
                uint64_t row_size_p1 = row_size_p2 + row_size%2;
               
                // column partition
                uint64_t col_size_p2 = col_size/2;
                uint64_t col_size_p1 = col_size_p2 + col_size%2;

                // k partition
                uint64_t k_size_p2 = k_size/2;
                uint64_t k_size_p1 = k_size_p2 + k_size%2;

                /* calculate the offsets requred for each submatrix */
                // k offsets
                uint64_t a_k_offset = k_size_p1*a.rows();
                uint64_t b_k_offset = k_size_p1;

                // row offsets
                uint64_t a_row_offset = row_size_p1;
                uint64_t c_row_offset = row_size_p1;

                // col offsets
                uint64_t c_col_offset = col_size_p1*c.rows();
                uint64_t b_col_offset = col_size_p1*b.rows();

                /* calculate the terms */
                // c_11 += a_11*b_11
                MatMultRecursive(a_raw_current,
                                 b_raw_current,
                                 c_raw_current,
                                 row_size_p1, col_size_p1, k_size_p1);

                // c_11 += a_12*b_21
                MatMultRecursive(a_raw_current + a_k_offset,
                                 b_raw_current + b_k_offset,
                                 c_raw_current,
                                 row_size_p1, col_size_p1, k_size_p2);

                // c_21 += a_21*b_11
                MatMultRecursive(a_raw_current + a_row_offset,
                                 b_raw_current,
                                 c_raw_current + c_row_offset,
                                 row_size_p2, col_size_p1, k_size_p1);

                // c_21 += a_22*b_21
                MatMultRecursive(a_raw_current + a_row_offset + a_k_offset,
                                 b_raw_current + b_k_offset,
                                 c_raw_current + c_row_offset, 
                                 row_size_p2, col_size_p1, k_size_p2);

                // c_12 += a_11*b_12
                MatMultRecursive(a_raw_current,
                                 b_raw_current + b_col_offset,
                                 c_raw_current + c_col_offset,
                                 row_size_p1, col_size_p2, k_size_p1);

                // c_12 += a_12*b_22
                MatMultRecursive(a_raw_current + a_k_offset,
                                 b_raw_current + b_col_offset + b_k_offset, 
                                 c_raw_current + c_col_offset,
                                 row_size_p1, col_size_p2, k_size_p2);

                // c_22 += a_21*b_12
                MatMultRecursive(a_raw_current + a_row_offset,
                                 b_raw_current + b_col_offset,
                                 c_raw_current + c_row_offset + c_col_offset,
                                 row_size_p2, col_size_p2, k_size_p1);

                // c_22 += a_22*b_22
                MatMultRecursive(a_raw_current + a_row_offset + a_k_offset,
                                 b_raw_current + b_col_offset + b_k_offset, 
                                 c_raw_current + c_row_offset + c_col_offset,
                                 row_size_p2, col_size_p2, k_size_p2);

            };

            // kickstart that recursion baby
            MatMultRecursive(a_raw, b_raw, c_raw,
                             a.rows(), b.cols(), a.cols());
        }

    } // namespace MatrixMultiply
} // namespace OptimizationTests
