
#ifndef THREEPP_QUATERNIONKEYFRAMETRACK_HPP
#define THREEPP_QUATERNIONKEYFRAMETRACK_HPP

#include "threepp/animation/KeyframeTrack.hpp"
#include "threepp/math/interpolants/QuaternionLinearInterpolant.hpp"

namespace threepp {

    class QuaternionKeyframeTrack: public KeyframeTrack {

    public:
        QuaternionKeyframeTrack(const std::string& name, const std::vector<float>& times, const std::vector<float>& values, const std::optional<Interpolation>& interpolation = {})
            : KeyframeTrack(name, times, values, sanitized(interpolation)) {}

        [[nodiscard]] std::string ValueTypeName() const override {

            return "quaternion";
        }

        std::unique_ptr<Interpolant> InterpolantFactoryMethodLinear(const Sample& times, const Sample& values, int valueSize, Sample* result) override {

            return std::make_unique<QuaternionLinearInterpolant>(times, values, valueSize, result);
        }

    private:
        // Which interpolation modes make sense for a rotation:
        //
        //   Discrete — fine. It copies a keyframe verbatim, so the quaternion
        //              stays exactly as authored. This is glTF's STEP, and the
        //              track used to force it to Linear, which is why STEP
        //              rotation animations played smoothly instead of snapping.
        //   Linear   — routes through QuaternionLinearInterpolant (slerp) above.
        //   Smooth   — NOT safe: a cubic evaluates the four components
        //              independently and denormalises the rotation. Coerced to
        //              Linear.
        //
        // Unspecified defaults to Linear rather than KeyframeTrack's global
        // default, since a rotation should slerp unless told otherwise.
        static std::optional<Interpolation> sanitized(const std::optional<Interpolation>& interpolation) {

            if (interpolation == Interpolation::Discrete) return Interpolation::Discrete;

            return Interpolation::Linear;
        }
    };

}// namespace threepp


#endif//THREEPP_QUATERNIONKEYFRAMETRACK_HPP
