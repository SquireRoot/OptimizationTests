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
            Timer() : _has_started(false), _min(std::numeric_limits<double>::max()), _max(0.0), _sum(0.0), _count(0.0) {}

            void Start() {
                _t0 = std::chrono::steady_clock::now();

                if (_has_started) throw std::logic_error("Cannot start an already started timer");
                _has_started = true;
            }

            double Split() {
                std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - _t0;
                
                if (!_has_started) throw std::logic_error("Cannot split a timer before it has been started");

                return dt.count();
            }

            double Stop() {
                std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - _t0;
                
                if (!_has_started) throw std::logic_error("Cannot stop a timer before it has been started");
                _has_started = false;

                double dt_ms = dt.count();

                _min = dt_ms < _min ? dt_ms : _min;
                _max = dt_ms > _max ? dt_ms : _max;
                _sum += dt_ms;
                _count++;

                return dt_ms;
            }

            void Stats(double& min, double& max, double& mean) {
                if (_count <= 0) {
                    throw std::runtime_error("Start() and Stop() must be called at least once before calling Stats()");
                }

                min = _min;
                max = _max;
                mean = _sum/_count;
            }

            std::string StatsString() {
                if (_count <= 0) {
                    throw std::runtime_error("Start() and Stop() must be called at least once before calling Stats()");
                }

                std::stringstream msg;
                msg << "min: " << _min << ", mean: " << _sum/_count << ", max: " << _max;
                return msg.str(); 
            }

            void Reset() {
                _min = std::numeric_limits<double>::max();
                _max = 0.0;
                _sum = 0.0;
                _count = 0;
            }

        private:
            bool _has_started;
            std::chrono::steady_clock::time_point _t0;

            double _min;
            double _max;
            double _sum;
            int _count;
        };

    } // namespace Util
} // namespace OptimizationTests

#endif // TIMER_H