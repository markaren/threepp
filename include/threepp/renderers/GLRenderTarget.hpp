// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderTarget.js

#ifndef THREEPPGLRENDERTARGETHPP
#define THREEPPGLRENDERTARGETHPP

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/textures/Texture.hpp"
#include "threepp/textures/DepthTexture.hpp"

#include "threepp/math/Vector4.hpp"

#include "threepp/utils/uuid.hpp"

#include <optional>

namespace threepp {

    class GLRenderTarget : public EventDispatcher {

    public:

        struct Options {

            int mapping;
            int wrapS;
            int wrapT;
            int magFilter;
            int minFilter = LinearFilter;
            int format;
            int type;
            int anisotropy;
            int encoding;

            bool generateMipmaps = false;
            bool depthBuffer = true;
            bool stencilBuffer = false;
            std::shared_ptr<DepthTexture> depthTexture;
        };

        const std::string uuid = utils::generateUUID();

        unsigned int width;
        unsigned int height;
        unsigned int depth = 1;

        Vector4 scissor{};
        bool scissorTest = false;

        Vector4 viewport{};

        std::shared_ptr<Texture> texture;

        bool depthBuffer;
        bool stencilBuffer;
        std::shared_ptr<DepthTexture> depthTexture;

        void setTexture(const std::shared_ptr<Texture> &texture) {

            texture->image = Image{width, height, depth};

            this->texture = texture;
        }

        void setSize(unsigned int width, unsigned int height, unsigned int depth = 1) {

            if (this->width != width || this->height != height || this->depth != depth) {

                this->width = width;
                this->height = height;
                this->depth = depth;

                this->texture->image->width = width;
                this->texture->image->height = height;
                this->texture->image->depth = depth;

                this->dispose();
            }

            this->viewport.set(0, 0, (float) width, (float) height);
            this->scissor.set(0, 0, (float) width, (float) height);
        }

        GLRenderTarget &copy(const GLRenderTarget &source) {

            this->width = source.width;
            this->height = source.height;
            this->depth = source.depth;

            this->viewport.copy(source.viewport);

            this->texture = source.texture;
            //                this->texture.image = { ...this->texture.image }; // See #20328.

            this->depthBuffer = source.depthBuffer;
            this->stencilBuffer = source.stencilBuffer;
            this->depthTexture = source.depthTexture;

            return *this;
        }

        void dispose() {

            this->dispatchEvent("dispose", this);
        }

        static std::shared_ptr<GLRenderTarget> create(unsigned int width, unsigned int height, const Options &options) {

            return std::shared_ptr<GLRenderTarget>(new GLRenderTarget(width, height, options));
        }

    protected:
        GLRenderTarget(unsigned int width, unsigned int height, const Options &options)
                : width(width), height(height),
                  scissor(0.f, 0.f, (float) width, (float) height),
                  viewport(0.f, 0.f, (float) width, (float) height),
                  depthBuffer(options.depthBuffer), stencilBuffer(options.stencilBuffer), depthTexture(options.depthTexture),
                  texture(Texture::create(std::nullopt, options.mapping, options.wrapS, options.wrapT, options.magFilter, options.minFilter, options.type, options.anisotropy, options.encoding)) {
        }
    };

}// namespace threepp

#endif//THREEPPGLRENDERTARGETHPP
