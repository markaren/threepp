// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderTarget.js

#ifndef THREEPP_GLRENDERTARGET_HPP
#define THREEPP_GLRENDERTARGET_HPP

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/textures/Texture.hpp"

#include "threepp/math/Vector4.hpp"

#include <optional>

namespace threepp {

    class GLRenderTarget: public EventDispatcher {

    public:
        struct Options {

            std::optional<Mapping> mapping;
            std::optional<TextureWrapping> wrapS;
            std::optional<TextureWrapping> wrapT;
            std::optional<Filter> magFilter;
            std::optional<Filter> minFilter;
            std::optional<Format> format;
            std::optional<Type> type;
            std::optional<int> anisotropy;
            std::optional<Encoding> encoding;

            bool generateMipmaps{false};
            bool depthBuffer{true};
            bool stencilBuffer{false};

            Options() = default;
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

        GLRenderTarget(unsigned int width, unsigned int height, const Options& options);

        GLRenderTarget(GLRenderTarget&&) = delete;
        GLRenderTarget(const GLRenderTarget&) = delete;
        GLRenderTarget& operator=(GLRenderTarget&&) = delete;
        GLRenderTarget& operator=(const GLRenderTarget&) = delete;

        void setSize(unsigned int width, unsigned int height, unsigned int depth = 1);

        GLRenderTarget& copy(const GLRenderTarget& source);

        void dispose();

        static std::unique_ptr<GLRenderTarget> create(unsigned int width, unsigned int height, const Options& options);

        ~GLRenderTarget() override;

    protected:
        bool disposed = false;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERTARGET_HPP
