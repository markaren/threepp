
#ifndef THREEPP_LINEARINTERPOLANT_HPP
#define THREEPP_LINEARINTERPOLANT_HPP

#include "threepp/math/Interpolant.hpp"

namespace threepp {

    class LinearInterpolant: public Interpolant {

    public:
        template<typename... Args>
        explicit LinearInterpolant(Args&&... args): Interpolant(std::forward<Args>(args)...) {}

        Sample interpolate_(size_t i1, float t0, float t, float t1) override {
            auto result = this->resultBuffer;
            auto values = this->sampleValues;
            auto stride = this->valueSize;

            auto offset1 = i1 * stride;
            auto offset0 = offset1 - stride;

            auto weight1 = (t - t0) / (t1 - t0);
            auto weight0 = 1 - weight1;

            for (auto i = 0; i != stride; ++i) {

                result->at(i) =
                        values[offset0 + i] * weight0 +
                        values[offset1 + i] * weight1;
            }

            return *result;
        }
    };

}// namespace threepp

#endif//THREEPP_LINEARINTERPOLANT_HPP
