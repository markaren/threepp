// https://github.com/mrdoob/three.js/blob/r129/src/helpers/SpotLightHelper.js

#ifndef THREEPP_SPOTLIGHTHELPER_HPP
#define THREEPP_SPOTLIGHTHELPER_HPP

#include <utility>

#include "threepp/lights/SpotLight.hpp"

namespace threepp {

    class LineSegments;

    class SpotLightHelper: public Object3D {

    public:
        void update();

        ~SpotLightHelper() override;

        static std::shared_ptr<SpotLightHelper> create(const std::shared_ptr<SpotLight>& light, std::optional<unsigned int> color = std::nullopt) {

            return std::shared_ptr<SpotLightHelper>(new SpotLightHelper(light, color));
        }

    protected:
        std::shared_ptr<SpotLight> light;
        std::optional<Color> color;

        std::shared_ptr<LineSegments> cone;

        SpotLightHelper(std::shared_ptr<SpotLight> light, std::optional<Color> color);
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHTHELPER_HPP
