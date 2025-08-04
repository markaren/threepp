// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBackground.js

#ifndef THREEPP_GLBACKGROUND_HPP
#define THREEPP_GLBACKGROUND_HPP

#include "threepp/constants.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/scenes/Scene.hpp"

namespace threepp {

    class Mesh;
    class GLRenderer;

    namespace gl {

        struct GLState;
        class GLObjects;
        class GLCubeMaps;
        struct GLRenderList;

        struct GLBackground {

            GLBackground(GLRenderer& renderer, GLCubeMaps& cubemaps, GLState& state, GLObjects& objects, bool premultipliedAlpha);

            void render(GLRenderList& renderList, Object3D* scene);

            [[nodiscard]] const Color& getClearColor() const;

            void setClearColor(const Color& color, float alpha = 1);

            [[nodiscard]] float getClearAlpha() const;

            void setClearAlpha(float alpha);


        private:
            GLRenderer& renderer;
            GLCubeMaps& cubemaps;
            GLState& state;
            GLObjects& objects;

            bool premultipliedAlpha;

            Color clearColor = Color(0x000000);
            float clearAlpha = 0;

            std::unique_ptr<Mesh> boxMesh = nullptr;

            Background* currentBackground = nullptr;
            unsigned int currentBackgroundVersion = 0;
            std::optional<ToneMapping> currentTonemapping;

            void setClear(const Color& color, float alpha);
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLBACKGROUND_HPP
