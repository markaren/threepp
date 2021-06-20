// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLightShadow.js

#ifndef THREEPP_POINTLIGHTSHADOWS_HPP
#define THREEPP_POINTLIGHTSHADOWS_HPP

#include "threepp/lights/LightShadow.hpp"

#include "threepp/cameras/Camera.hpp"

namespace threepp {

    class PointLightShadow : public LightShadow {

    public:
        PointLightShadow(std::shared_ptr<Camera> camera) : camera_(camera) {}

    private:
        std::shared_ptr<Camera> camera_;
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHTSHADOWS_HPP
