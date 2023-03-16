// https://github.com/mrdoob/three.js/tree/r129/src/core/Clock.js

#ifndef THREEPP_CLOCK_HPP
#define THREEPP_CLOCK_HPP

#include <memory>

namespace threepp {

    class Clock {

    public:
        explicit Clock(bool autoStart = true);

        void start();

        void stop();

        float getElapsedTime();

        float getDelta();

        ~Clock();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_CLOCK_HPP
