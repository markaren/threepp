
#ifndef THREEPP_FLOAT_VIEW_HPP
#define THREEPP_FLOAT_VIEW_HPP

#include <functional>
#include <optional>
#include <utility>

namespace threepp {

    class float_view {

    public:
        float value;

        float_view(float value = 0) : value(value) {}

        float operator()() const {

            return value;
        }

        float_view &operator=(float v) {
            this->value = v;
            if (f_) f_.value()();
            return *this;
        }

        float operator*(float f) const {
            return value * f;
        }

        float operator*(const float_view &f) const {
            return value * f.value;
        }

        float_view &operator*=(float f) {
            value *= f;
            if (f_) f_.value()();
            return *this;
        }

        float operator/(float f) const {
            return value / f;
        }

        float_view &operator/=(float f) {
            value /= f;
            if (f_) f_.value()();
            return *this;
        }

        float operator+(float f) const {
            return value + f;
        }

        float operator+(const float_view &f) const {
            return value + f.value;
        }

        float_view &operator+=(float f) {
            value += f;
            if (f_) f_.value()();
            return *this;
        }

        float operator-(float f) const {
            return value - f;
        }

        float operator-(const float_view &f) const {
            return value - f.value;
        }

        float_view &operator-=(float f) {
            value -= f;
            if (f_) f_.value()();
            return *this;
        }

        float_view &operator++() {
            value++;
            if (f_) f_.value()();
            return *this;
        }

        float_view &operator--() {
            value--;
            if (f_) f_.value()();
            return *this;
        }

    private:
        std::optional<std::function<void()>> f_;

        void setCallback(std::function<void()> f) {

            f_ = std::move(f);
        }

        friend class Euler;
    };

}// namespace threepp

#endif//THREEPP_FLOAT_VIEW_HPP
