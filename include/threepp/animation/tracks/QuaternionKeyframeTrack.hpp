
#ifndef THREEPP_QUATERNIONKEYFRAMETRACK_HPP
#define THREEPP_QUATERNIONKEYFRAMETRACK_HPP

#include "threepp/animation/KeyframeTrack.hpp"
#include "threepp/math/interpolants/QuaternionLinearInterpolant.hpp"

namespace threepp {

    class QuaternionKeyframeTrack: public KeyframeTrack {

    public:
        QuaternionKeyframeTrack(const std::string& name, const std::vector<float>& times, const std::vector<float>& values, const std::optional<Interpolation>& interpolation = {})
            : KeyframeTrack(name, times, values, Interpolation::Linear) {}

        [[nodiscard]] std::string ValueTypeName() const override {

            return "quaternion";
        }

        std::unique_ptr<Interpolant> InterpolantFactoryMethodLinear(const Sample& times, const Sample& values, int valueSize, Sample* result) override {

            return std::make_unique<QuaternionLinearInterpolant>(times, values, valueSize, result);
        }
    };

}// namespace threepp


#endif//THREEPP_QUATERNIONKEYFRAMETRACK_HPP
