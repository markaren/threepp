// https://github.com/mrdoob/three.js/blob/r129/src/helpers/HemisphereLightHelper.js

#ifndef THREEPP_HEMISPHERELIGHTHELPER_HPP
#define THREEPP_HEMISPHERELIGHTHELPER_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Color.hpp"

#include <memory>
#include <optional>

namespace threepp {

    class HemisphereLight;

    class HemisphereLightHelper: public Object3D {

    public:
        std::optional<Color> color;

        void update();

        void dispose();

        static std::shared_ptr<HemisphereLightHelper> create(HemisphereLight& light, float size, const std::optional<Color>& color = std::nullopt);

        ~HemisphereLightHelper() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        HemisphereLightHelper(HemisphereLight& light, float size, const std::optional<Color>& color);
    };

}// namespace threepp

#endif//THREEPP_HEMISPHERELIGHTHELPER_HPP
