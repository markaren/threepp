// https://github.com/mrdoob/three.js/blob/r129/src/helpers/SpotLightHelper.js

#ifndef THREEPP_SPOTLIGHTHELPER_HPP
#define THREEPP_SPOTLIGHTHELPER_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"

#include <memory>
#include <optional>

namespace threepp {

    class SpotLight;
    class LineSegments;

    class SpotLightHelper: public Object3D {

    public:
        void update();

        ~SpotLightHelper() override;

        static std::shared_ptr<SpotLightHelper> create(SpotLight& light, std::optional<Color> color = std::nullopt);

    private:
        SpotLight& light;
        std::optional<Color> color;

        std::shared_ptr<LineSegments> cone;

        SpotLightHelper(SpotLight& light, std::optional<Color> color);
    };

}// namespace threepp

#endif//THREEPP_SPOTLIGHTHELPER_HPP
