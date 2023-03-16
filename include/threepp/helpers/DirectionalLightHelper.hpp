// https://github.com/mrdoob/three.js/blob/r129/src/helpers/DirectionalLightHelper.js

#ifndef THREEPP_DIRECTIONALLIGHTHELPER_HPP
#define THREEPP_DIRECTIONALLIGHTHELPER_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/objects/Line.hpp"

namespace threepp {

    class DirectionalLightHelper: public Object3D {

    public:
        void update();

        static std::shared_ptr<DirectionalLightHelper> create(
                const std::shared_ptr<DirectionalLight>& light,
                float size = 1,
                std::optional<Color> color = std::nullopt);

    protected:
        float size;
        std::optional<Color> color;

        std::shared_ptr<DirectionalLight> light;
        std::shared_ptr<Line> lightPlane;
        std::shared_ptr<Line> targetLine;

        DirectionalLightHelper(std::shared_ptr<DirectionalLight> light, float size, std::optional<Color> color);
    };

}// namespace threepp

#endif//THREEPP_DIRECTIONALLIGHTHELPER_HPP
