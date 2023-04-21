// https://github.com/mrdoob/three.js/blob/r129/src/helpers/PointLightHelper.js

#ifndef THREEPP_POINTLIGHTHELPER_HPP
#define THREEPP_POINTLIGHTHELPER_HPP

#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/objects/Mesh.hpp"

#include <memory>
#include <optional>

namespace threepp {

    class PointLight;

    class PointLightHelper: public Mesh {

    public:
        void update();

        static std::shared_ptr<PointLightHelper> create(PointLight& light, float sphereSize, std::optional<Color> color = std::nullopt);

    private:
        std::optional<Color> color;
        PointLight& light;

        PointLightHelper(PointLight& light, float sphereSize, std::optional<Color> color);
    };

}// namespace threepp

#endif//THREEPP_POINTLIGHTHELPER_HPP
