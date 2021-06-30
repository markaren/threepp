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
        std::optional<Texture> environment;
        std::optional<FogVariant> fog;

        std::shared_ptr<Material> overrideMaterial;

        bool autoUpdate = true;

        std::string type() const override {
            return "Scene";
        }

        static std::shared_ptr<Scene> create() {
            return std::shared_ptr<Scene>(new Scene());
        }

    protected:
        Scene() = default;
    };

    struct EmptyScene : Scene {};

}// namespace threepp

#endif//THREEPP_SCENE_HPP
