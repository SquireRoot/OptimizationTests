/*
MatrixMultiplySimple.cpp
Evan Newman
*/

#include <eigen3/Eigen/Core>

namespace OptimizationTests {
    namespace MatrixMultiply {

        void MatMultSimple(const Eigen::Ref<const Eigen::MatrixXf> a,
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

        void MatMultSimpleOptimized(const Eigen::Ref<const Eigen::MatrixXf> a,
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

    } // namespace MatrixMultiply
} // namespace OptimizationTests