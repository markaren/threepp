
#ifndef THREEPP_DISCRETEINTERPOLANT_HPP
#define THREEPP_DISCRETEINTERPOLANT_HPP

#include "threepp/math/Interpolant.hpp"

namespace threepp {

    class DiscreteInterpolant: public Interpolant {

    public:
        template<typename... Args>
        explicit DiscreteInterpolant(Args&&... args): Interpolant(std::forward<Args>(args)...) {}

        Sample interpolate_(size_t i1, [[maybe_unused]] float t0, [[maybe_unused]] float t, [[maybe_unused]] float t1) override {

            return copySampleValue_(i1-1);
        }
    };

}// namespace threepp

#endif//THREEPP_DISCRETEINTERPOLANT_HPP
