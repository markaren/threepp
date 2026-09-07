
#ifndef THREEPP_FLOAT_VIEW_HPP
#define THREEPP_FLOAT_VIEW_HPP

#include <algorithm>
#include <functional>

namespace threepp {

    // An internal wrapper around float that notifies its owner when the value changes.
    //
    // float_view is implicitly convertible to float, so all non-mutating arithmetic,
    // comparison and stream operations are inherited for free from the built-in float
    // operators. Only the mutating operations (which notify) and clamp are defined here.
    //
    // The notification target is a pointer to a std::function owned by the parent
    // (Euler/Quaternion), so a float_view stores only a float plus a pointer rather than an
    // embedded std::function. Copying or moving a float_view transfers the value only: the
    // notification target belongs to the parent at its fixed address, so a detached copy is
    // intentionally left unwired (mutating it is a no-op rather than a dangling call).
    class float_view {

    public:
        float_view(float value = 0)
            : value_(value) {}

        float_view(const float_view& o)
            : value_(o.value_) {}

        float_view(float_view&& o) noexcept
            : value_(o.value_) {}

        // Structural assignment transfers the value but keeps each side wired to its own
        // parent; user-facing edits go through operator=(float), which notifies.
        float_view& operator=(const float_view& o) {

            value_ = o.value_;

            return *this;
        }

        float_view& operator=(float_view&& o) noexcept {

            value_ = o.value_;

            return *this;
        }

        operator float() const {

            return value_;
        }

        float_view& operator=(float v) {

            value_ = v;
            notify();

            return *this;
        }

        float_view& operator*=(float f) {

            value_ *= f;
            notify();

            return *this;
        }

        float_view& operator/=(float f) {

            value_ /= f;
            notify();

            return *this;
        }

        float_view& operator+=(float f) {

            value_ += f;
            notify();

            return *this;
        }

        float_view& operator-=(float f) {

            value_ -= f;
            notify();

            return *this;
        }

        float_view& operator++() {

            value_++;
            notify();

            return *this;
        }

        float_view& operator--() {

            value_--;
            notify();

            return *this;
        }

        constexpr float_view& clamp(float min, float max) {

            value_ = std::max(min, std::min(max, value_));

            return *this;
        }

        // Observe a callback owned elsewhere; the referent must outlive this float_view.
        void setCallback(const std::function<void()>& f) {

            onChange_ = &f;
        }
        void setCallback(std::function<void()>&&) = delete;// a temporary would dangle

    private:
        void notify() const {

            if (onChange_ && *onChange_) (*onChange_)();
        }

        float value_;
        const std::function<void()>* onChange_ = nullptr;

        friend class Euler;
        friend class Quaternion;
    };

}// namespace threepp

#endif//THREEPP_FLOAT_VIEW_HPP
