
#ifndef THREEPP_KEYFRAMETRACK_HPP
#define THREEPP_KEYFRAMETRACK_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "threepp/constants.hpp"
#include "threepp/math/Interpolant.hpp"
#include "threepp/math/interpolants/CubicInterpolant.hpp"
#include "threepp/math/interpolants/DiscreteInterpolant.hpp"
#include "threepp/math/interpolants/LinearInterpolant.hpp"

#include <functional>

namespace threepp {

    class Interpolant;

    class KeyframeTrack {

        using InterpolantFactory = std::function<std::unique_ptr<Interpolant>(const Sample&, const Sample&, int, Sample*)>;

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

        size_t getValueSize() const;

        void setInterpolation(Interpolation interpolation);

        KeyframeTrack& shift(float timeOffset);

        KeyframeTrack& scale(float timeScale);

        KeyframeTrack& trim(float startTime, float endTime);

        KeyframeTrack& optimize();

        std::unique_ptr<Interpolant> createInterpolant(std::vector<float>* result) const;

        virtual ~KeyframeTrack() = default;

    protected:
        KeyframeTrack(std::string name,
                      const std::vector<float>& times,
                      const std::vector<float>& values,
                      const std::optional<Interpolation>& interpolation = {});

    private:
        std::string name_;
        std::vector<float> times_;
        std::vector<float> values_;
        Interpolation interpolation_;

        std::function<std::unique_ptr<Interpolant>(const Sample&, const Sample&, int, Sample*)> createInterpolant_;

        virtual std::unique_ptr<Interpolant> InterpolantFactoryMethodDiscrete(const Sample& times, const Sample& values, int valueSize, Sample* result) {

            return std::make_unique<DiscreteInterpolant>(times, values, valueSize, result);
        }

        virtual std::unique_ptr<Interpolant> InterpolantFactoryMethodLinear(const Sample& times, const Sample& values, int valueSize, Sample* result) {

            return std::make_unique<LinearInterpolant>(times, values, valueSize, result);
        }

        virtual std::unique_ptr<Interpolant> InterpolantFactoryMethodSmooth(const Sample& times, const Sample& values, int valueSize, Sample* result) {

            return std::make_unique<CubicInterpolant>(times, values, valueSize, result);
        }
    };


}// namespace threepp

#endif//THREEPP_KEYFRAMETRACK_HPP
