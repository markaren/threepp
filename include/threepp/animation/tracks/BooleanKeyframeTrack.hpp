
#ifndef THREEPP_BOOLEANKEYFRAMETRACK_HPP
#define THREEPP_BOOLEANKEYFRAMETRACK_HPP

#include "threepp/animation/KeyframeTrack.hpp"

namespace threepp {

    class BooleanKeyframeTrack : public KeyframeTrack {

    public:
        // Booleans always use discrete interpolation
        BooleanKeyframeTrack(const std::string& name,
                             const std::vector<float>& times,
                             const std::vector<float>& values)
            : KeyframeTrack(name, times, values, Interpolation::Discrete) {}

        [[nodiscard]] std::string ValueTypeName() const override { return "bool"; }

        static std::shared_ptr<BooleanKeyframeTrack> create(
                const std::string& name,
                const std::vector<float>& times,
                const std::vector<float>& values) {
            return std::make_shared<BooleanKeyframeTrack>(name, times, values);
        }
    };

}// namespace threepp

#endif//THREEPP_BOOLEANKEYFRAMETRACK_HPP
