// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBackground.js

#ifndef THREEPP_GLBACKGROUND_HPP
#define THREEPP_GLBACKGROUND_HPP

#include "GLState.hpp"

#include "threepp/math/Color.hpp"

#include "threepp/scenes/Scene.hpp"

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLBackground {

            GLBackground(GLState &state, bool premultipliedAlpha);

            void render(GLRenderer &renderer, Object3D *scene);

            [[nodiscard]] const Color &getClearColor() const;

            void setClearColor(const Color &color, float alpha = 1);

            [[nodiscard]] float getClearAlpha() const;

            void setClearAlpha(float alpha);


        private:
            GLState &state;

            bool premultipliedAlpha;

            Color clearColor = Color(0x000000);
            float clearAlpha = 0;

            void setClear(const Color &color, float alpha);
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLBACKGROUND_HPP
