// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Fog.js

#ifndef THREEPP_FOG_HPP
#define THREEPP_FOG_HPP

#include "threepp/math/Color.hpp"

namespace threepp {

    class Fog {

    public:
        Color color;
        float nearPlane;
        float farPlane;

        explicit Fog(const Color& color, float _near = 1, float _far = 1000);

        bool operator==(const Fog& f) const;
    };

}// namespace threepp

#endif//THREEPP_FOG_HPP
