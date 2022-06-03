/*
MatrixMultiply.cpp
Evan Newman
*/

#include "MatrixMultiply.h"

// System 
#include <stdexcept>
#include <iostream>
#include <cstdint>

// Libraries
#include <Core> // Eigen stuff

// Local
#include "MatrixMultiplySimple.h"
#include "MatrixMultiplyTiled.h"
#include "MatrixMultiplyCacheOblivious.h"

#include "Util/Timer.h"

namespace OptimizationTests {
    namespace MatrixMultiply {

        void RunMatrixMultiplyTests() {
            std::cout << "-------- MatrixMultiply Tests --------\n";

            uint64_t dim1 = 750; // rows of c and a
            uint64_t dim2 = 750; // columns of c and b
            uint64_t dim3 = 750; // columns of a and rows of b

            const int num_iter = 1;

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
                a.operator*(b);
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
                std::cout << label << ":  " << dt << "ms, result ";

                // test output
                if (c.isApprox(c_eigen)) {
                    std::cout << "ok\n";
                } else {
                    std::cout << "error!\n";
                }
            };

            /* ----- Test the functions ----- */
            RunTest(MatMultSimple, "MatMultSimple");
            RunTest(MatMultSimpleOptimized, "MatMultSimpleOptimized");

            RunTest(MatMultTiled, "MatMultTiled");
            RunTest(MatMultTiledOptimized, "MatMultTiledOptimized");

            RunTest(MatMultCacheOblivious, "MatMultCacheOblivious");
            RunTest(MatMultCacheObliviousOptimized, "MatMultCacheObliviousOptimized");
        }
        
    } // namespace MatrixMultiply
} // namepsace OptimizationTests