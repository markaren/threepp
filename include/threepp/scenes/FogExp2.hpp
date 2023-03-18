// https://github.com/mrdoob/three.js/blob/r129/src/scenes/FogExp2.js

#ifndef THREEPP_FOGEXP2_HPP
#define THREEPP_FOGEXP2_HPP

#include "threepp/math/Color.hpp"

namespace threepp {

    class FogExp2 {

    public:
        Color color;
        float density;

        explicit FogExp2(const Color& hex, float density = 0.00025f);

        bool operator==(const FogExp2& f) const;
    };

}// namespace threepp

#endif//THREEPP_FOGEXP2_HPP
