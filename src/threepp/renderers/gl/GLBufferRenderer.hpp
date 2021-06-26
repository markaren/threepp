// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBufferRenderer.js

#ifndef THREEPP_GLBUFFERRENDERER_HPP
#define THREEPP_GLBUFFERRENDERER_HPP

#include "GLCapabilities.hpp"
#include "GLInfo.hpp"

namespace threepp::gl {

    struct BufferRenderer {

        explicit BufferRenderer(GLInfo &info) : info_(info) {}

        void setMode(GLenum mode) {

            this->mode_ = mode;
        }

        virtual void render(GLint start, GLsizei count) = 0;

        virtual ~BufferRenderer() = default;

    protected:
        GLenum mode_;
        GLInfo &info_;
    };

    struct GLBufferRenderer : BufferRenderer {

        explicit GLBufferRenderer(GLInfo &info) : BufferRenderer(info) {}

        void render(GLint start, GLsizei count) override {

            glDrawArrays(mode_, start, count);

            info_.update(count, mode_, 1);
        }

        void renderInstances(GLint start, GLsizei count, GLsizei primcount) {

            if (primcount == 0) return;

            glDrawArraysInstanced(mode_, start, count, primcount);

            info_.update(count, mode_, primcount);
        }
    };

    struct GLIndexedBufferRenderer : BufferRenderer {

        GLIndexedBufferRenderer(GLInfo &info)
            : BufferRenderer(info){}

        void setIndex(Buffer &value) {

            type_ = value.type;
            bytesPerElement_ = value.bytesPerElement;
        }

        void render(GLint start, GLsizei count) override {

            glDrawElements(mode_, count, type_, reinterpret_cast<const void *>(start * bytesPerElement_));
        }

        void renderInstances(GLint start, GLsizei count, GLsizei primcount) {

            if (primcount == 0) return;

            glDrawElementsInstanced(mode_, count, type_, reinterpret_cast<const void *>(start * bytesPerElement_), primcount);

            info_.update(count, mode_, primcount);
        }


    private:
        GLint type_;
        GLsizei bytesPerElement_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLBUFFERRENDERER_HPP
