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

            Eigen::MatrixXf a(100, 100);
            Eigen::MatrixXf b(100, 100);
            Eigen::MatrixXf c(100, 100);

            a.setRandom();
            b.setRandom();
            c.setZero();

            Util::Timer timer;
            
            // run and time eigen 
            timer.Start();
            Eigen::MatrixXf c_eigen = (a*b).eval();
            double time_eigen = timer.Stop();
            std::cout << "Eigen:   \t" << time_eigen << "ms\n";

            // run and time MatMult1
            timer.Start();
            MatMult1(a, b, c);
            double time_matmult1 = timer.Stop();
            std::cout << "MatMult1:\t" << time_matmult1 << "ms, result ";

            // test output of MatMult1
            if (((c - c_eigen).array().abs() > 1e-5).any()) {
                std::cout << "error!\n";
            } else {
                std::cout << "ok\n";
            }
        }

        void MatMult1(const Eigen::Ref<const Eigen::MatrixXf> a,
                      const Eigen::Ref<const Eigen::MatrixXf> b,
                      Eigen::Ref<Eigen::MatrixXf> c) {

            if (a.rows() != c.rows()
               || a.cols() != b.rows()
               || b.cols() != c.cols()) {
                   throw std::invalid_argument("matrices a, b, and c have incompatible sizes");
            }

            const float* a_raw = a.data();
            const float* b_raw = b.data();
            float* c_raw = c.data();

            for (int c_row = 0; c_row < c.rows(); c_row++) {
                for (int c_col = 0; c_col < c.cols(); c_col++) {
                    float sum = 0;

                    for (int i = 0; i < a.cols(); i++) {
                        sum += a_raw[c_row + i*a.rows()]*b_raw[i + c_col*b.rows()];
                    }

                    c_raw[c_row + c_col*c.rows()] = sum;
                }
            }
        }


    } // namespace MatrixMultiply

} // namepsace OptimizationTests