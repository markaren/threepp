
#ifndef THREEPP_LIMIT_HPP
#define THREEPP_LIMIT_HPP

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

        bool clamp(float &value) const {
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
