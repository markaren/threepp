// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderTarget.js

#ifndef THREEPP_GLRENDERTARGET_HPP
#define THREEPP_GLRENDERTARGET_HPP

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/textures/Texture.hpp"

#include "threepp/math/Vector4.hpp"

#include <optional>

namespace threepp {

    class GLRenderTarget: public EventDispatcher {

        struct Options {

            int mapping;
            int wrapS;
            int wrapT;
            int magFilter;
            int minFilter = LinearFilter;
            int type;
            int anisotropy;
            int encoding;

            bool generateMipmaps = false;
            bool depthBuffer = true;
            bool stencilBuffer = false;
            std::optional<Texture> depthTexture;
        };

    public:
        GLRenderTarget(unsigned int width, unsigned int height, const Options &options)
            : width_(width), height_(height),
              scissor_(0, 0, (float) width, (float) height),
              viewPort_(0, 0, (float) width, (float) height),
              depthBuffer_(options.depthBuffer), stencilBuffer_(options.stencilBuffer), depthTexture_(options.depthTexture),
              texture_(std::nullopt, options.mapping, options.wrapS, options.wrapT, options.magFilter, options.minFilter, options.type, options.anisotropy, options.encoding) {
        }

        void setTexture(Texture &texture) {

            texture.image = Image {width_, height_, depth_};

            this->texture_ = texture;
        }
        
        void setSize( unsigned int width,  unsigned int height,  unsigned int depth = 1 ) {

            if ( this->width != width || this->height != height || this->depth != depth ) {

                this->width = width;
                this->height = height;
                this->depth = depth;

                this->texture.image.width = width;
                this->texture.image.height = height;
                this->texture.image.depth = depth;

                this->dispose();

            }

            this->viewport_.set( 0, 0, (float) width, (float) height );
            this->scissor_.set( 0, 0, (float) width, (float) height );

        }

        GLRenderTarget &copy( const GLRenderTarget &source ) {

                this->width = source.width;
                this->height = source.height;
                this->depth = source.depth;

                this->viewport_.copy( source.viewport );

                this->texture_ = source.texture;
//                this->texture_.image = { ...this->texture.image }; // See #20328.

                this->depthBuffer_ = source.depthBuffer;
                this->stencilBuffer_ = source.stencilBuffer;
                this->depthTexture_ = source.depthTexture;

                return this;

        }
        
        void dispose() {
            
            this->dispatchEvent("dispose");
        }

    private:
        const unsigned int width_;
        const unsigned int height_;
        const unsigned int depth_ = 1;

        Vector4 scissor_;
        bool scissorTest_ = false;

        Vector4 viewPort_;

        Texture texture_;

        bool depthBuffer_;
        bool stencilBuffer_;
        std::optional<Texture> depthTexture_;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERTARGET_HPP
