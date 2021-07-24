// https://github.com/mrdoob/three.js/blob/r129/src/lights/Light.js

#ifndef THREEPP_LIGHT_HPP
#define THREEPP_LIGHT_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"

#include <optional>

namespace threepp {

    class Light : public Object3D {

    public:
        Color color;
        float intensity;

        Light(const Light &) = delete;

        [[nodiscard]] virtual std::string type() const {

            return "Light";
        }

        virtual void dispose(){};

    protected:
        template<class T>
        Light(T color, std::optional<float> intensity)
            : color(color), intensity(intensity.value_or(1)) {}

    };

}// namespace threepp

#endif//THREEPP_LIGHT_HPP
