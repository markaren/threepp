// https://github.com/mrdoob/three.js/tree/r129/src/core/Clock.js

#ifndef THREEPP_CLOCK_HPP
#define THREEPP_CLOCK_HPP

#include <chrono>

namespace threepp {

    class clock {

    public:
        explicit clock(bool autoStart = true): autoStart_(autoStart){}

        void start() {

            startTime_ = std::chrono::system_clock::now();

            oldTime_ = startTime_;
            elapsedTime_ = 0;
            running_ = true;

        }

        void stop() {

            getElapsedTime();
            running_ = false;
            autoStart_ = false;

        }

        double getElapsedTime() {

            getDelta();
            return elapsedTime_;

        }

        double getDelta() {

            double diff = 0;

            if (autoStart_ && !running_) {

                start();
                return 0;

            }

            if (running_) {

                const auto newTime = std::chrono::system_clock::now();

                diff = std::chrono::duration_cast<std::chrono::microseconds>(newTime - oldTime_).count() / 1000000.0;
                oldTime_ = newTime;

                elapsedTime_ += diff;

            }

            return diff;

        }

    private:
        bool autoStart_;
        bool running_ = false;

        double elapsedTime_;

        std::chrono::time_point<std::chrono::system_clock> startTime_;
        std::chrono::time_point<std::chrono::system_clock> oldTime_;

    };

}

#endif //THREEPP_CLOCK_HPP
