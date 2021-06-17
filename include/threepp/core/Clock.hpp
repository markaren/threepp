// https://github.com/mrdoob/three.js/tree/r129/src/core/Clock.js

#ifndef THREEPP_CLOCK_HPP
#define THREEPP_CLOCK_HPP

#include <chrono>

namespace threepp {

    class Clock {

    public:
        explicit Clock(bool autoStart = true) : autoStart_(autoStart) {}

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

        float getElapsedTime() {

            getDelta();
            return elapsedTime_;
        }

        float getDelta() {

            float diff = 0;

            if (autoStart_ && !running_) {

                start();
                return 0;
            }

            if (running_) {

                const auto newTime = std::chrono::system_clock::now();

                diff = std::chrono::duration_cast<std::chrono::microseconds>(newTime - oldTime_).count() / 1000000.0f;
                oldTime_ = newTime;

                elapsedTime_ += diff;
            }

            return diff;
        }

    private:
        bool autoStart_;
        bool running_ = false;

        float elapsedTime_ = 0;

        std::chrono::time_point<std::chrono::system_clock> startTime_;
        std::chrono::time_point<std::chrono::system_clock> oldTime_;
    };

}// namespace threepp

#endif//THREEPP_CLOCK_HPP
