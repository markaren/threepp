
#ifndef THREEPP_STRINGKEYFRAMETRACK_HPP
#define THREEPP_STRINGKEYFRAMETRACK_HPP

#include "threepp/animation/KeyframeTrack.hpp"

namespace threepp {

    class StringKeyframeTrack : public KeyframeTrack {

    public:
        // Strings always use discrete interpolation
        StringKeyframeTrack(const std::string& name,
                            const std::vector<float>& times,
                            const std::vector<float>& values)
            : KeyframeTrack(name, times, values, Interpolation::Discrete) {}

        [[nodiscard]] std::string ValueTypeName() const override { return "string"; }

        static std::shared_ptr<StringKeyframeTrack> create(
                const std::string& name,
                const std::vector<float>& times,
                const std::vector<float>& values) {
            return std::make_shared<StringKeyframeTrack>(name, times, values);
        }
    };

}// namespace threepp

#endif//THREEPP_STRINGKEYFRAMETRACK_HPP
