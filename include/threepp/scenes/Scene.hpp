// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"

#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"

#include "threepp/textures/Texture.hpp"

#include <memory>
#include <variant>

namespace threepp {

    typedef std::variant<Fog, FogExp2> FogVariant;

    class Scene : public Object3D {

    public:
        std::optional<Color> background;
        std::shared_ptr<Texture> environment;
        std::optional<FogVariant> fog;

        std::shared_ptr<Material> overrideMaterial;

        bool autoUpdate = true;

        static std::shared_ptr<Scene> create() {

            return std::make_shared<Scene>();
        }
    };

    inline bool operator==(const FogVariant &f1, const FogVariant &f2) {

        if (f1.index() != f2.index()) return false;

        if (f1.index() == 0) {
            return std::get<0>(f1) == std::get<0>(f2);
        } else {
            return std::get<1>(f1) == std::get<1>(f2);
        }
    }

    inline bool operator!=(const FogVariant &f1, const FogVariant &f2) {

        return !(f1 == f2);
    }

}// namespace threepp

#endif//THREEPP_SCENE_HPP
