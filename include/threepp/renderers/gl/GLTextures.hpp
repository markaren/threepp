// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLTextures.js

#ifndef THREEPP_GLTEXTURES_HPP
#define THREEPP_GLTEXTURES_HPP

#include "GLCapabilities.hpp"
#include "GLInfo.hpp"
#include "GLProperties.hpp"
#include "GLState.hpp"

#include "GLUniforms.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/textures/Texture.hpp"


#include <memory>
#include <unordered_map>

namespace threepp::gl {

    struct GLTextures {

        const int maxTextures = GLCapabilities::instance().maxTextures;
        const int maxCubemapSize = GLCapabilities::instance().maxCubemapSize;
        const int maxTextureSize = GLCapabilities::instance().maxTextureSize;
        const int maxSamples = GLCapabilities::instance().maxSamples;

        GLTextures(
                GLState &state,
                GLProperties &properties,
                GLInfo &info);

        void generateMipmap(unsigned int target, const Texture &texture, unsigned int width, unsigned int height);

        void setTextureParameters(unsigned int textureType, Texture &texture);

        void initTexture(TextureProperties &textureProperties, Texture &texture);

        void uploadTexture(TextureProperties &textureProperties, Texture &texture, unsigned int slot);

        void uploadCubeTexture(TextureProperties &textureProperties, Texture &texture, unsigned int slot);

        void deallocateTexture(Texture *texture);

        void deallocateRenderTarget(GLRenderTarget *renderTarget);

        void resetTextureUnits();

        int allocateTextureUnit();

        void setTexture2D(Texture &texture, unsigned int slot);

        void setTexture2DArray(Texture &texture, unsigned int slot);

        void setTexture3D(Texture &texture, unsigned int slot);

        void setTextureCube(Texture &texture, unsigned int slot);

        // Setup storage for target texture and bind it to correct framebuffer
        void setupFrameBufferTexture(unsigned int framebuffer, const std::shared_ptr<GLRenderTarget> &renderTarget, Texture &texture, unsigned int attachment, unsigned int textureTarget);

        void setupRenderBufferStorage(unsigned int renderbuffer, const std::shared_ptr<GLRenderTarget> &renderTarget);

        // Setup resources for a Depth Texture for a FBO (needs an extension)
        void setupDepthTexture(unsigned int framebuffer, const std::shared_ptr<GLRenderTarget> &renderTarget);

        // Setup GL resources for a non-texture depth buffer
        void setupDepthRenderbuffer(const std::shared_ptr<GLRenderTarget> &renderTarget);

        // Set up GL resources for the render target
        void setupRenderTarget(const std::shared_ptr<GLRenderTarget> &renderTarget);

        void updateRenderTargetMipmap(const std::shared_ptr<GLRenderTarget> &renderTarget);

    private:
        struct TextureEventListener : EventListener {

            explicit TextureEventListener(GLTextures &scope) : scope_(scope) {}

            void onEvent(Event &event) override;

        private:
            GLTextures &scope_;
        };

        struct RenderTargetEventListener : EventListener {

            explicit RenderTargetEventListener(GLTextures &scope) : scope_(scope) {}

            void onEvent(Event &event) override;

        private:
            GLTextures &scope_;
        };

        GLState &state;
        GLProperties &properties;
        GLInfo &info;

        TextureEventListener onTextureDispose_;
        RenderTargetEventListener onRenderTargetDispose_;

        int textureUnits = 0;
    };

}// namespace threepp::gl

#endif//THREEPP_GLTEXTURES_HPP
