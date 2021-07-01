// https://github.com/mrdoob/three.js/blob/r129/src/lights/PointLightShadow.js

#ifndef THREEPP_POINTLIGHTSHADOWS_HPP
#define THREEPP_POINTLIGHTSHADOWS_HPP

#include "threepp/lights/LightShadow.hpp"

namespace threepp {

    class PointLight;

    class PointLightShadow : public LightShadow {

    public:

        explicit PointLightShadow();

        void updateMatrices(PointLight *light, int viewportIndex = 0);

    private:
        std::vector<Vector4> _viewports;
        std::vector<Vector3> _cubeDirections;
        std::vector<Vector3> _cubeUps;
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHTSHADOWS_HPP
