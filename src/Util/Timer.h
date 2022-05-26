/*
Timer.h a timer which allows for start, split and stop
Evan Newman
*/

#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <stdexcept>

namespace OptimizationTests {
    namespace Util {

        class Timer {
        public:
            Timer() : has_started_(false) {}

            void Start() {
                t0_ = std::chrono::steady_clock::now();

                if (has_started_) throw std::logic_error("Cannot start an already started timer");
                has_started_ = true;
            }

            double Split() {
                std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0_;
                
                if (!has_started_) throw std::logic_error("Cannot split a timer before it has been started");

                return dt.count();
            }

            double Stop() {
                std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0_;
                
                if (!has_started_) throw std::logic_error("Cannot stop a timer before it has been started");
                has_started_ = false;

                return dt.count();
            }

        private:
            bool has_started_;
            std::chrono::steady_clock::time_point t0_;
        };

    } // namespace Util
} // namespace OptimizationTests

#endif // TIMER_H