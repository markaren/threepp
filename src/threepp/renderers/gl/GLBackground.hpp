// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBackground.js

#ifndef THREEPP_GLBACKGROUND_HPP
#define THREEPP_GLBACKGROUND_HPP

#include "threepp/math/Color.hpp"
#include "threepp/objects/Mesh.hpp"

#include "threepp/scenes/Scene.hpp"

#include "threepp/renderers/gl/GLState.hpp"

namespace threepp::gl {

    struct GLBackground {

        GLBackground(GLState &state, bool premultipliedAlpha) : state(state), premultipliedAlpha(premultipliedAlpha) {}

        void render() {
        }

        [[nodiscard]] Color getClearColor() const {
            return clearColor;
        }

        void setClearColor(const Color &color, float alpha = 1) {

            clearColor.copy( color );
            clearAlpha = alpha;
            setClear( clearColor, clearAlpha );
        }

        [[nodiscard]] float getClearAlpha() const {
            return clearAlpha;
        }

        void setClearAlpha(float alpha) {

            clearAlpha = alpha;
            setClear( clearColor, clearAlpha );
        }


    private:
        GLState &state;

        bool premultipliedAlpha;

        Color clearColor = Color(0x000000);
        float clearAlpha = 0;

        std::shared_ptr<Mesh> planeMesh = nullptr;
        std::shared_ptr<Mesh> boxMesh = nullptr;

        std::any currentBackground;
        unsigned int currentBackgroundVersion = 0;
        //        let currentTonemapping = null;

        void setClear(const Color &color, float alpha) {

            state.colorBuffer.setClear(color.r, color.g, color.b, alpha, premultipliedAlpha);
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLBACKGROUND_HPP
