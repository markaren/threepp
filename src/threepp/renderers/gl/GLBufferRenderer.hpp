// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBufferRenderer.js

#ifndef THREEPP_GLBUFFERRENDERER_HPP
#define THREEPP_GLBUFFERRENDERER_HPP

#include "GLCapabilities.hpp"
#include "GLInfo.hpp"

namespace threepp::gl {

    struct GLBufferRenderer {

        GLBufferRenderer(GLInfo &info) : info_(info) {}

        void setMode(int mode) {

            this->mode_ = mode;
        }

        void render(int start, int count) {

            glDrawArrays(mode_, start, count);

            info_.update(count, mode_, 1);
        }

        void renderInstances(int start, int count, int primcount) {

            if (primcount == 0) return;

            glDrawArraysInstanced(mode_, start, count, primcount);

            info_.update(count, mode_, primcount);
        }

    private:
        int mode_;
        GLInfo &info_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLBUFFERRENDERER_HPP
