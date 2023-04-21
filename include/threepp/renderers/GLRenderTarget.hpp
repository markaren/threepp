// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderTarget.js

#ifndef THREEPPGLRENDERTARGETHPP
#define THREEPPGLRENDERTARGETHPP

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/textures/Texture.hpp"

#include "threepp/math/Vector4.hpp"

#include <optional>

namespace threepp {

    class DepthTexture;

    class GLRenderTarget: public EventDispatcher {

    public:
        struct Options {

            std::optional<int> mapping;
            std::optional<int> wrapS;
            std::optional<int> wrapT;
            std::optional<int> magFilter;
            std::optional<int> minFilter;
            std::optional<int> format;
            std::optional<int> type;
            std::optional<int> anisotropy;
            std::optional<int> encoding;

            bool generateMipmaps = false;
            bool depthBuffer = true;
            bool stencilBuffer = false;
            std::shared_ptr<DepthTexture> depthTexture;
        };

        const std::string uuid;

        unsigned int width;
        unsigned int height;
        unsigned int depth = 1;

        Vector4 scissor;
        bool scissorTest = false;

        Vector4 viewport;

        std::shared_ptr<Texture> texture;

        bool depthBuffer;
        bool stencilBuffer;
        std::shared_ptr<DepthTexture> depthTexture;

        GLRenderTarget(const GLRenderTarget&) = delete;
        GLRenderTarget(const GLRenderTarget&&) = delete;
        GLRenderTarget operator=(const GLRenderTarget&) = delete;

        void setTexture(const std::shared_ptr<Texture>& tex);

        void setSize(unsigned int width, unsigned int height, unsigned int depth = 1);

        GLRenderTarget& copy(const GLRenderTarget& source);

        void dispose();

        static std::shared_ptr<GLRenderTarget> create(unsigned int width, unsigned int height, const Options& options);

        ~GLRenderTarget() override;

    protected:
        bool disposed = false;
        GLRenderTarget(unsigned int width, unsigned int height, const Options& options);
    };

}// namespace threepp

#endif//THREEPPGLRENDERTARGETHPP
