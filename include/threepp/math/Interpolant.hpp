
#ifndef THREEPP_INTERPOLANT_HPP
#define THREEPP_INTERPOLANT_HPP

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "threepp/constants.hpp"

namespace threepp {

    struct InterpolantSettings {
        Ending endingStart{Ending::ZeroCurvature};
        Ending endingEnd{Ending::ZeroCurvature};
    };


    typedef std::vector<float> Sample;

    class Interpolant {

    public:
        InterpolantSettings* settings;

        Interpolant(
                Sample parameterPositions,
                Sample sampleValues,
                int sampleSize,
                Sample* resultBuffer);

        Sample evaluate(float t);

        Sample copySampleValue_(size_t index) const;

        virtual ~Interpolant() = default;

    protected:
        size_t _cachedIndex{0};
        Sample parameterPositions;
        Sample* resultBuffer;
        Sample sampleValues;
        int valueSize;

        std::optional<InterpolantSettings> DefaultSettings_;

        [[nodiscard]] std::optional<InterpolantSettings> getSettings_() const;

        virtual Sample interpolate_(size_t i1, float t0, float t, float t1) = 0;

        virtual void intervalChanged_(size_t i1, float t0, float t1){};

    private:
        Sample _resultBuffer;

        friend class AnimationAction;
        friend class AnimationMixer;
    };

}// namespace threepp

#endif//THREEPP_INTERPOLANT_HPP
