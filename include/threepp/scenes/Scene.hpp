// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"

#include <memory>

namespace threepp {

    class Scene: public Object3D {

    public:

        bool autoUpdate = true;

        Scene(const Scene&) = delete;

        std::string type() const override {
            return "Scene";
        }

        static std::shared_ptr<Scene> create() {
            return std::shared_ptr<Scene>(new Scene());
        }

    protected:
        Scene() = default;

    };

}

#endif//THREEPP_SCENE_HPP
