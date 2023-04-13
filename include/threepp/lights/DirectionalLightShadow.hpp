// https://github.com/mrdoob/three.js/blob/r129/src/lights/DirectionalLightShadow.js

#ifndef THREEPP_DIRECTIONALLIGHTSHADOW_HPP
#define THREEPP_DIRECTIONALLIGHTSHADOW_HPP

#include "threepp/lights/LightShadow.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"

namespace threepp {

    class DirectionalLightShadow: public LightShadow {

    public:
        static std::shared_ptr<DirectionalLightShadow> create() {

            return std::shared_ptr<DirectionalLightShadow>(new DirectionalLightShadow());
        }

    protected:
        DirectionalLightShadow()
            : LightShadow(OrthographicCamera::create(-5, 5, 5, -5, 0.5f, 500)) {}
    };

}// namespace threepp

#endif//THREEPP_DIRECTIONALLIGHTSHADOW_HPP
