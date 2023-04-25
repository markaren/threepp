
#include "threepp/core/Clock.hpp"

#include <chrono>

using namespace threepp;


struct Clock::Impl {

    Clock& scope;
    std::chrono::time_point<std::chrono::system_clock> startTime_;
    std::chrono::time_point<std::chrono::system_clock> oldTime_;

    explicit Impl(Clock& scope): scope(scope) {}

    void start() {

        startTime_ = std::chrono::system_clock::now();

        oldTime_ = startTime_;
        scope.elapsedTime = 0;
        scope.running = true;
    }

    void stop() {

        getElapsedTime();
        scope.running = false;
        scope.autoStart = false;
    }

    float getElapsedTime() {

        getDelta();
        return scope.elapsedTime;
    }

    float getDelta() {

        float diff = 0;

        if (scope.autoStart && !scope.running) {

            start();
            return 0;
        }

        if (scope.running) {

            const auto newTime = std::chrono::system_clock::now();

            diff = std::chrono::duration_cast<std::chrono::microseconds>(newTime - oldTime_).count() / 1000000.0f;
            oldTime_ = newTime;

            scope.elapsedTime += diff;
        }

        return diff;
    }
};

Clock::Clock(bool autoStart)
    : autoStart(autoStart),
      pimpl_(std::make_unique<Impl>(*this)) {}


void Clock::start() {

    pimpl_->start();
}

void Clock::stop() {

    pimpl_->stop();
}

float Clock::getElapsedTime() {

    return pimpl_->getElapsedTime();
}

float Clock::getDelta() {

    return pimpl_->getDelta();
}

Clock::~Clock() = default;
