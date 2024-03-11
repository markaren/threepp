
#ifndef THREEPP_QUATERNIONKEYFRAMETRACK_HPP
#define THREEPP_QUATERNIONKEYFRAMETRACK_HPP

#include "threepp/animation/KeyframeTrack.hpp"

namespace threepp {

    class QuaternionKeyframeTrack: public KeyframeTrack {

    public:
        QuaternionKeyframeTrack(const std::string& name, const std::vector<float>& times, const std::vector<float>& values, const std::optional<Interpolation>& interpolation = {})
            : KeyframeTrack(name + ".quaternion", times, values, Interpolation::Linear) {}

        [[nodiscard]] std::string ValueTypeName() const override {

            return "quaternion";
        }
    };

}// namespace threepp


#endif//THREEPP_QUATERNIONKEYFRAMETRACK_HPP
