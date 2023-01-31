// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLBufferRenderer.js

#ifndef THREEPP_GLBUFFERRENDERER_HPP
#define THREEPP_GLBUFFERRENDERER_HPP

#include "GLAttributes.hpp"
#include "GLCapabilities.hpp"
#include "GLInfo.hpp"

namespace threepp::gl {

    struct BufferRenderer {

        explicit BufferRenderer(GLInfo &info) : info_(info) {}

        void setMode(unsigned int mode);

        virtual void render(int start, int count) = 0;

        virtual void renderInstances(int start, int count, int primcount) = 0;

        virtual ~BufferRenderer() = default;

    protected:
        GLInfo &info_;
        unsigned int mode_;
    };

    struct GLBufferRenderer : BufferRenderer {

        explicit GLBufferRenderer(GLInfo &info)
            : BufferRenderer(info) {}

        void render(int start, int count) override;

        void renderInstances(int start, int count, int primcount) override;
    };

    struct GLIndexedBufferRenderer : BufferRenderer {

        explicit GLIndexedBufferRenderer(GLInfo &info)
            : BufferRenderer(info) {}

        void setIndex(const Buffer &value);

        void render(int start, int count) override;

        void renderInstances(int start, int count, int primcount) override;


    private:
        int type_{};
        size_t bytesPerElement_{};
    };

}// namespace threepp::gl

#endif//THREEPP_GLBUFFERRENDERER_HPP
