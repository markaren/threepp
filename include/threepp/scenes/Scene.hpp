// https://github.com/mrdoob/three.js/blob/r129/src/scenes/Scene.js

#ifndef THREEPP_SCENE_HPP
#define THREEPP_SCENE_HPP

#include "threepp/core/Object3D.hpp"

#include <optional>

namespace threepp {

    class Scene: public Object3D {

    public:

        std::string type = "Scene";

        bool autoUpdate = true;

        Scene()

    };

}

#endif//THREEPP_SCENE_HPP
