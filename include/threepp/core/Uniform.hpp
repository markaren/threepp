// https://github.com/mrdoob/three.js/blob/r129/src/core/Uniform.js

#ifndef THREEPP_UNIFORM_HPP
#define THREEPP_UNIFORM_HPP

#include <any>
#include <optional>
#include <utility>

namespace threepp {

    using NULL_UNIFORM = std::any;

    class Uniform {

    public:
        std::optional<bool> needsUpdate;

        explicit Uniform(std::any value = NULL_UNIFORM()) : value_(std::move(value)) {}

        template<class T>
        [[nodiscard]] T &value() {
            return std::any_cast<T &>(value_);
        }

        void value(std::any value) {
            this->value_ = std::move(value);
        }

    private:
        std::any value_;
    };

}// namespace threepp

#endif//THREEPP_UNIFORM_HPP
