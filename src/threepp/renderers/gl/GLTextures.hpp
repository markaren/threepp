// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLTextures.js

#ifndef THREEPP_GLTEXTURES_HPP
#define THREEPP_GLTEXTURES_HPP

#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLProperties.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include "threepp/renderers/GLRenderTarget.hpp"


namespace threepp::gl {

    struct GLTextures {

        const int maxTextures;
        const int maxCubemapSize;
        const int maxTextureSize;
        const int maxSamples;

        GLTextures(GLState& state, GLProperties& properties, GLInfo& info);

        void generateMipmap(unsigned int target, Texture& texture, unsigned int width, unsigned int height);

        void setTextureParameters(unsigned int textureType, Texture& texture);

        void initTexture(TextureProperties* textureProperties, Texture& texture);

        void uploadTexture(TextureProperties* textureProperties, Texture& texture, unsigned int slot);

        void uploadCubeTexture(TextureProperties* textureProperties, Texture& texture, unsigned int slot);

        void deallocateTexture(Texture* texture);

        void deallocateRenderTarget(GLRenderTarget* renderTarget);

        void resetTextureUnits();

        int allocateTextureUnit();

        void setTexture2D(Texture& texture, unsigned int slot);

        void setTexture2DArray(Texture& texture, unsigned int slot);

        void setTexture3D(Texture& texture, unsigned int slot);

        void setTextureCube(Texture& texture, unsigned int slot);

        // Setup storage for target texture and bind it to correct framebuffer
        void setupFrameBufferTexture(unsigned int framebuffer, GLRenderTarget* renderTarget, Texture& texture, unsigned int attachment, unsigned int textureTarget);

        void setupRenderBufferStorage(unsigned int renderbuffer, GLRenderTarget* renderTarget);

        void setupDepthTexture(unsigned int framebuffer, GLRenderTarget* renderTarget);

        // Setup GL resources for a non-texture depth buffer
        void setupDepthRenderbuffer(GLRenderTarget* renderTarget);

        // Set up GL resources for the render target
        void setupRenderTarget(GLRenderTarget* renderTarget);

        void updateRenderTargetMipmap(GLRenderTarget* renderTarget);

        [[nodiscard]] std::optional<unsigned int> getGlTexture(Texture& texture) const;

    private:
        struct TextureEventListener: EventListener {

            explicit TextureEventListener(GLTextures* scope): scope_(scope) {}

            void onEvent(Event& event) override;

        private:
            GLTextures* scope_;
        };

        struct RenderTargetEventListener: EventListener {

            explicit RenderTargetEventListener(GLTextures* scope): scope_(scope) {}

            void onEvent(Event& event) override;

        private:
            GLTextures* scope_;
        };

        GLInfo* info;
        GLState* state;
        GLProperties* properties;

        TextureEventListener onTextureDispose_;
        RenderTargetEventListener onRenderTargetDispose_;

        int textureUnits = 0;
    };

}// namespace threepp::gl

#endif//THREEPP_GLTEXTURES_HPP
