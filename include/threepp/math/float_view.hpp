
#ifndef THREEPP_FLOAT_VIEW_HPP
#define THREEPP_FLOAT_VIEW_HPP

#include <functional>
#include <optional>
#include <ostream>
#include <utility>

namespace threepp {

    class float_view {

    public:
        float_view(float value = 0)
            : value_(value) {}

        inline float operator()() const {

            return value_;
        }

        inline float_view& operator=(float v) {

            this->value_ = v;
            if (f_) f_.value()();

            return *this;
        }

        inline float operator*(float f) const {

            return value_ * f;
        }

        inline float operator*(const float_view& f) const {

            return value_ * f.value_;
        }

        inline float_view& operator*=(float f) {

            value_ *= f;
            if (f_) f_.value()();

            return *this;
        }

        inline float operator/(float f) const {

            return value_ / f;
        }

        inline float_view& operator/=(float f) {

            value_ /= f;
            if (f_) f_.value()();

            return *this;
        }

        inline float operator+(float f) const {

            return value_ + f;
        }

        inline float operator+(const float_view& f) const {

            return value_ + f.value_;
        }

        inline float_view& operator+=(float f) {

            value_ += f;
            if (f_) f_.value()();

            return *this;
        }

        inline float operator-(float f) const {

            return value_ - f;
        }

        inline float operator-(const float_view& f) const {

            return value_ - f.value_;
        }

        inline float_view& operator-=(float f) {

            value_ -= f;
            if (f_) f_.value()();

            return *this;
        }

        inline float_view& operator++() {

            value_++;
            if (f_) f_.value()();

            return *this;
        }

        inline float_view& operator--() {

            value_--;
            if (f_) f_.value()();

            return *this;
        }

        inline bool operator==(float other) const {

            return value_ == other;
        }

        inline bool operator!=(float other) const {

            return value_ != other;
        }

        inline bool operator==(const float_view& other) const {

            return value_ == other.value_;
        }

        inline bool operator!=(const float_view& other) const {

            return value_ != other.value_;
        }

        inline constexpr float_view& clamp(float min, float max) {

            value_ = std::max(min, std::min(max, value_));

            return *this;
        }

        void setCallback(std::function<void()> f) {

            f_ = std::move(f);
        }

        friend std::ostream& operator<<(std::ostream& os, const float_view& f) {
            os << f.value_;
            return os;
        }

    private:
        float value_;
        std::optional<std::function<void()>> f_;

        friend class Euler;
        friend class Quaternion;
    };

}// namespace threepp

#endif//THREEPP_FLOAT_VIEW_HPP
