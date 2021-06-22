// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBackground.js

#ifndef THREEPP_GLBACKGROUND_HPP
#define THREEPP_GLBACKGROUND_HPP

#include "threepp/math/Color.hpp"
#include "threepp/objects/Mesh.hpp"

#include "threepp/scenes/Scene.hpp"

namespace threepp::gl {

    struct GLBackground {

        void render() {

        }

    private:
        Color clearColor = Color(0x000000);
        float clearAlpha = 0;

        std::shared_ptr<Mesh> planeMesh = nullptr;
        std::shared_ptr<Mesh> boxMesh = nullptr;

        std::any currentBackground;
        unsigned int currentBackgroundVersion = 0;
//        let currentTonemapping = null;

    };

}

#endif//THREEPP_GLBACKGROUND_HPP
