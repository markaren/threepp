//

#ifndef THREEPP_DIRECTIONALLIGHTSHADOW_HPP
#define THREEPP_DIRECTIONALLIGHTSHADOW_HPP

#include "threepp/lights/LightShadow.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"

namespace threepp {

    class DirectionalLightShadow: public LightShadow {

        DirectionalLightShadow()
            : LightShadow(OrthographicCamera::create(-5, 5, 5, -5, 0.5f, 500)) {}

    };

}

#endif//THREEPP_DIRECTIONALLIGHTSHADOW_HPP
