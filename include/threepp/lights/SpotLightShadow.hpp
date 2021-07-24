// https://github.com/mrdoob/three.js/blob/r129/src/lights/SpotLightShadow.js

#ifndef THREEPP_SPOTLIGHTSHADOW_HPP
#define THREEPP_SPOTLIGHTSHADOW_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/lights/LightShadow.hpp"

namespace threepp {

    class SpotLight;

    class SpotLightShadow : public LightShadow {

    public:
        float focus = 1;

        void updateMatrices(SpotLight *light);

        static std::shared_ptr<SpotLightShadow> create() {

            return std::shared_ptr<SpotLightShadow>(new SpotLightShadow());
        }

    protected:
        SpotLightShadow()
            : LightShadow(PerspectiveCamera::create(50, 1, 0.5f, 500)) {}
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHTSHADOW_HPP
