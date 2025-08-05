
#ifndef VECTORKEYFRAMETRACK_HPP
#define VECTORKEYFRAMETRACK_HPP

#include "threepp/animation/KeyframeTrack.hpp"

#include <vector>

namespace threepp {

    class VectorKeyframeTrack: public KeyframeTrack {

    public:
        VectorKeyframeTrack(const std::string& name, const std::vector<float>& times, const std::vector<float>& values, const std::optional<Interpolation>& interpolation = {})
            : KeyframeTrack(name, times, values) {}

        [[nodiscard]] std::string ValueTypeName() const override {

            return "vector";
        }

    };

}// namespace threepp


#endif //VECTORKEYFRAMETRACK_HPP
