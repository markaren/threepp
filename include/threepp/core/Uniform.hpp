// https://github.com/mrdoob/three.js/blob/r129/src/core/Uniform.js

#ifndef THREEPP_UNIFORM_HPP
#define THREEPP_UNIFORM_HPP

#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/math/Vector4.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    typedef std::variant<int, float, Color, Vector2, Vector3> NestedUniformValue;
    typedef std::variant<bool, int, float, Color, Vector2, Vector3, Vector4, Matrix3, Matrix4, Matrix4*, std::shared_ptr<Texture>, std::vector<float>, std::vector<Vector2>, std::vector<Vector3>, std::vector<Matrix3>, std::vector<Matrix4>, std::vector<std::shared_ptr<Texture>>, std::unordered_map<std::string, NestedUniformValue>, std::vector<std::unordered_map<std::string, NestedUniformValue>*>> UniformValue;

    class Uniform {

    public:
        std::optional<bool> needsUpdate;

        explicit Uniform(std::optional<UniformValue> value = std::nullopt, std::optional<bool> needsUpdate = std::nullopt)
            : value_(std::move(value)), needsUpdate(needsUpdate) {}

        [[nodiscard]] bool hasValue() const {
            return value_.has_value();
        }

        UniformValue &value() {
            return *value_;
        }

        template<class T>
        [[nodiscard]] T &value() {
            if (!value_.has_value()) value_ = T();
            return std::get<T>(*value_);
        }

        void setValue(UniformValue value) {
            this->value_ = std::move(value);
        }

        void setValue(std::reference_wrapper<UniformValue> value) {
            this->value_ = value.get();
        }

    private:
        std::optional<UniformValue> value_;
    };

    typedef std::unordered_map<std::string, Uniform> UniformMap;

}// namespace threepp

#endif//THREEPP_UNIFORM_HPP
