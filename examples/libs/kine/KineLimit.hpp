
#ifndef THREEPP_LIMIT_HPP
#define THREEPP_LIMIT_HPP

#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace kine {

    class KineLimit {

    public:
        KineLimit(std::optional<float> min = std::nullopt, std::optional<float> max = std::nullopt)
            : min_(min), max_(max) {}

        [[nodiscard]] std::optional<float> min() const {
            return min_;
        }

        [[nodiscard]] std::optional<float> max() const {
            return max_;
        }

        [[nodiscard]] float mean() const {

            float lower = min_.value_or(-std::numeric_limits<float>::max());
            float upper = max_.value_or(std::numeric_limits<float>::max());

            return (lower + upper) / 2.f;
        }

        [[nodiscard]] float normalize(float value) const {
            clampWithinLimit(value);
            return threepp::math::mapLinear(value, min_.value_or(-std::numeric_limits<float>::max()), max_.value_or(std::numeric_limits<float>::max()), 0, 1);
        }

        [[nodiscard]] float denormalize(float value) const {
            return threepp::math::mapLinear(std::clamp(value, 0.f, 1.f), 0, 1, min_.value_or(-std::numeric_limits<float>::max()), max_.value_or(std::numeric_limits<float>::max()));
        }

        bool clampWithinLimit(float& value) const {
            if (std::isnan(value)) value = mean();
            if (min_ && *min_ > value) {
                value = *min_;
                return true;
            } else if (max_ && *max_ < value) {
                value = *max_;
                return true;
            }
            return false;
        }

    private:
        std::optional<float> min_;
        std::optional<float> max_;
    };

}// namespace kine

#endif//THREEPP_LIMIT_HPP
