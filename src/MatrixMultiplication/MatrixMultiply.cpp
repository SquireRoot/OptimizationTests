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
#include <eigen3/Eigen/Core> // Eigen stuff
// #include <blis/cblas.h>

// Local
#include "MatrixMultiplySimple.h"
#include "MatrixMultiplyTiled.h"
#include "MatrixMultiplyCacheOblivious.h"
#include "MatrixMultiplyFastest.h"

#include "Util/Timer.h"

namespace OptimizationTests {
    namespace MatrixMultiply {

        void RunMatrixMultiplyTests() {
            std::cout << "-------- MatrixMultiply Tests --------" << std::endl
                      << "Function Name, Min (ms), Mean (ms), Max (ms)" << std::endl;

            uint64_t dim1 = 750; // rows of c and a
            uint64_t dim2 = 750; // columns of c and b
            uint64_t dim3 = 750; // columns of a and rows of b

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

            for (int i = 0; i < num_iter; i++) {
                timer.Start();
                c_eigen = (a*b).eval();
                timer.Stop();
            }
            std::cout << "Eigen: " << timer.StatsString() << std::endl;

            auto RunTest = [&](auto func, std::string label) {
                c.setZero();

                double min = std::numeric_limits<double>::max();
                double max = 0;
                double mean = 0;

                timer.Reset();
                for (int i = 0; i < num_iter; i++) {
                    timer.Start();
                    func(a, b, c);
                    timer.Stop();
                }

                std::cout << label << ": " << timer.StatsString() << ", result ";

                // test output
                if (c.isApprox(c_eigen)) {
                    std::cout << "ok";
                } else {
                    std::cout << "error!";
                }

                std::cout << std::endl;
            };

            /* ----- Test the functions ----- */
            RunTest(MatMultSimple, "MatMultSimple");
            RunTest(MatMultSimpleOptimized, "MatMultSimpleOptimized");

            RunTest(MatMultTiled, "MatMultTiled");
            RunTest(MatMultTiledOptimized, "MatMultTiledOptimized");

            // RunTest(MatMultCacheOblivious, "MatMultCacheOblivious");
            // RunTest(MatMultCacheObliviousOptimized, "MatMultCacheObliviousOptimized");

            // RunTest(MatMultFastest, "MatMultFastest");
        }
        
    } // namespace MatrixMultiply
} // namepsace OptimizationTests