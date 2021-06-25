// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderTarget.js

#ifndef THREEPP_GLRENDERTARGET_HPP
#define THREEPP_GLRENDERTARGET_HPP

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/textures/Texture.hpp"

#include "threepp/math/Vector4.hpp"

namespace threepp {

    class GLRenderTarget: public EventDispatcher {

        struct Options {

            int mapping;
            int wrapS;
            int wrapT;
            int magFilter;
            int minFilter;
            int type;
            int anisotropy;
            int encoding;
        };

    public:
        GLRenderTarget(unsigned int width, unsigned int height, const Options &options)
            : width_(width), height_(height),
              scissor(0, 0, (float) width, (float) height),
              viewPort(0, 0, (float) width, (float) height),
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

            this->viewport.set( 0, 0, width, height );
            this->scissor.set( 0, 0, width, height );

        }

    private:
        const unsigned int width_;
        const unsigned int height_;
        const unsigned int depth_ = 1;

        Vector4 scissor;
        bool scissorTest = false;

        Vector4 viewPort;

        Texture texture_;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERTARGET_HPP
