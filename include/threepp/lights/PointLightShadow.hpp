// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLightShadow.js

#ifndef THREEPP_POINTLIGHTSHADOWS_HPP
#define THREEPP_POINTLIGHTSHADOWS_HPP

#include "threepp/lights/LightShadow.hpp"

namespace threepp {

    class PointLight;

    class PointLightShadow : public LightShadow {

    public:
        void updateMatrices(PointLight *light, int viewportIndex = 0);

        static std::shared_ptr<PointLightShadow> create() {

            return std::shared_ptr<PointLightShadow>(new PointLightShadow());
        }

    protected:
        std::vector<Vector3> _cubeDirections;
        std::vector<Vector3> _cubeUps;

        PointLightShadow();
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHTSHADOWS_HPP
