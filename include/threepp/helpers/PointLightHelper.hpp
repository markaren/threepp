// https://github.com/mrdoob/three.js/blob/r129/src/helpers/PointLightHelper.js

#ifndef THREEPP_POINTLIGHTHELPER_HPP
#define THREEPP_POINTLIGHTHELPER_HPP

#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class PointLightHelper: public Mesh {

    public:
        void update();

        static std::shared_ptr<PointLightHelper> create(const std::shared_ptr<PointLight>& light, float sphereSize, std::optional<unsigned int> color = std::nullopt);

    protected:
        std::optional<Color> color;
        std::shared_ptr<PointLight> light;

        PointLightHelper(std::shared_ptr<PointLight> light, float sphereSize, std::optional<Color> color);
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHTHELPER_HPP
