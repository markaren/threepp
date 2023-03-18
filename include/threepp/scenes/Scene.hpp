// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"

#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"

#include <memory>
#include <variant>

namespace threepp {

    class Texture;
    typedef std::variant<Fog, FogExp2> FogVariant;

    class Scene: public Object3D {

    public:
        std::optional<Color> background;
        std::shared_ptr<Texture> environment;
        std::optional<FogVariant> fog;

        std::shared_ptr<Material> overrideMaterial;

        bool autoUpdate = true;

        static std::shared_ptr<Scene> create();
    };

}// namespace threepp

#endif//THREEPP_SCENE_HPP
