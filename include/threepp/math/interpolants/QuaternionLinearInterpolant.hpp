
#ifndef THREEPP_QUATERNIONLINEARINTERPOLANT_HPP
#define THREEPP_QUATERNIONLINEARINTERPOLANT_HPP

#include "threepp/math/Interpolant.hpp"
#include "threepp/math/Quaternion.hpp"

namespace threepp {

    class QuaternionLinearInterpolant: public Interpolant {

    public:
        template<typename... Args>
        explicit QuaternionLinearInterpolant(Args&&... args): Interpolant(std::forward<Args>(args)...) {}

        Sample interpolate_(size_t i1, float t0, float t, float t1) override {
            const auto result = this->resultBuffer;
            const auto& values = this->sampleValues;
            const auto stride = this->valueSize;

            const auto alpha = (t - t0) / (t1 - t0);

            auto offset = i1 * stride;

            for (const auto end = offset + stride; offset != end; offset += 4) {

                Quaternion::slerpFlat(*result, 0, values, offset - stride, values, offset, alpha);
            }

            return *result;
        }
    };

}// namespace threepp

#endif//THREEPP_QUATERNIONLINEARINTERPOLANT_HPP
