// https://github.com/mrdoob/three.js/blob/r129/src/lights/SpotLightShadow.js

#ifndef THREEPP_SPOTLIGHTSHADOW_HPP
#define THREEPP_SPOTLIGHTSHADOW_HPP

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/lights/LightShadow.hpp"

namespace threepp {

    class SpotLightShadow: public LightShadow {

    public:
        float focus = 1;

        void updateMatrices(Light* light) override;

        static std::shared_ptr<SpotLightShadow> create();

    protected:
        SpotLightShadow();
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHTSHADOW_HPP
