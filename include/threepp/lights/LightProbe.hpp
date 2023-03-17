// https://github.com/mrdoob/three.js/blob/r129/src/lights/LightProbe.js

#ifndef THREEPP_LIGHTPROBE_HPP
#define THREEPP_LIGHTPROBE_HPP

#include <utility>

#include "threepp/lights/Light.hpp"

#include "threepp/math/SphericalHarmonics3.hpp"

namespace threepp {

    class LightProbe: public Light {

    public:
        SphericalHarmonis3 sh;

    protected:
        explicit LightProbe(SphericalHarmonis3 sh = SphericalHarmonis3(), float intensity = 1)
            : Light(0xffffff, intensity), sh(std::move(sh)) {}
    };

}// namespace threepp

#endif//THREEPP_LIGHTPROBE_HPP
