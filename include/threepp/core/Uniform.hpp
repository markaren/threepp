// https://github.com/mrdoob/three.js/blob/r129/src/core/Uniform.js

#ifndef THREEPP_UNIFORM_HPP
#define THREEPP_UNIFORM_HPP

#include <any>
#include <utility>

namespace threepp {

    class Uniform {

    public:
        explicit Uniform(std::any value) : value_(std::move(value)) {}

        [[nodiscard]] std::any &value() {
            return value_;
        }

    private:
        std::any value_;
    };

}// namespace threepp

#endif//THREEPP_UNIFORM_HPP
