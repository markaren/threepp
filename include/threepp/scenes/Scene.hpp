// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"

#include "threepp/scenes/Fog.hpp"

#include <memory>

namespace threepp {

    class Scene : public Object3D {

    public:
        std::optional<Color> background;
        std::optional<Texture> environment;
        std::optional<Fog> fog;

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
