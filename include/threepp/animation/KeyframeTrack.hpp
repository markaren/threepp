
#ifndef THREEPP_KEYFRAMETRACK_HPP
#define THREEPP_KEYFRAMETRACK_HPP

#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "threepp/animation/AnimationUtils.hpp"
#include "threepp/constants.hpp"

namespace threepp {

    class Interpolant;

    class KeyframeTrack {

    public:
        static Interpolation defaultInterpolation;

        [[nodiscard]] virtual std::string ValueTypeName() const = 0;

        [[nodiscard]] std::string getName() const {

            return name_;
        }

        [[nodiscard]] const std::vector<float>& getTimes() const {

            return times_;
        }

        [[nodiscard]] const std::vector<float>& getValues() const {

            return values_;
        }

        [[nodiscard]] Interpolation getInterpolation() const {

            return interpolation_;
        }

        size_t getValueSize() {

            return values_.size() / times_.size();
        }

        void setInterpolation(Interpolation interpolation);

        KeyframeTrack& shift(float timeOffset);

        KeyframeTrack& scale(float timeScale);

        KeyframeTrack& trim(float startTime, float endTime);

        KeyframeTrack& optimize();

        std::unique_ptr<Interpolant> createInterpolant(std::vector<float>* result);

        virtual ~KeyframeTrack() = default;

    protected:
        KeyframeTrack(std::string name,
                      const std::vector<float>& times,
                      const std::vector<float>& values,
                      std::optional<Interpolation> interpolation);

    private:
        std::string name_;
        std::vector<float> times_;
        std::vector<float> values_;
        Interpolation interpolation_;
    };


}// namespace threepp

#endif//THREEPP_KEYFRAMETRACK_HPP
