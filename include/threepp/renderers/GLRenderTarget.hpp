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
              scissor_(0.f, 0.f, (float) width, (float) height),
              viewport_(0.f, 0.f, (float) width, (float) height),
              depthBuffer_(options.depthBuffer), stencilBuffer_(options.stencilBuffer), depthTexture_(options.depthTexture),
              texture_(std::nullopt, options.mapping, options.wrapS, options.wrapT, options.magFilter, options.minFilter, options.type, options.anisotropy, options.encoding) {
        }

        void setTexture(Texture &texture) {

            texture.image = Image {width_, height_, depth_};

            this->texture_ = texture;
        }
        
        void setSize( unsigned int width,  unsigned int height,  unsigned int depth = 1 ) {

            if ( this->width_ != width || this->height_ != height || this->depth_ != depth ) {

                this->width_ = width;
                this->height_ = height;
                this->depth_ = depth;

                this->texture_.image->width = width;
                this->texture_.image->height = height;
                this->texture_.image->depth = depth;

                this->dispose();

            }

            this->viewport_.set( 0, 0, (float) width, (float) height );
            this->scissor_.set( 0, 0, (float) width, (float) height );

        }

        GLRenderTarget &copy( const GLRenderTarget &source ) {

                this->width_ = source.width_;
                this->height_ = source.height_;
                this->depth_ = source.depth_;

                this->viewport_.copy( source.viewport_ );

                this->texture_ = source.texture_;
//                this->texture_.image = { ...this->texture.image }; // See #20328.

                this->depthBuffer_ = source.depthBuffer_;
                this->stencilBuffer_ = source.stencilBuffer_;
                this->depthTexture_ = source.depthTexture_;

                return *this;

        }
        
        void dispose() {
            
            this->dispatchEvent("dispose");
        }

    private:
        unsigned int width_;
        unsigned int height_;
        unsigned int depth_ = 1;

        Vector4 scissor_;
        bool scissorTest_ = false;

        Vector4 viewport_;

        Texture texture_;

        bool depthBuffer_;
        bool stencilBuffer_;
        std::optional<Texture> depthTexture_;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERTARGET_HPP
