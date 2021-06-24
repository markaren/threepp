// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLTextures.js

#ifndef THREEPP_GLTEXTURES_HPP
#define THREEPP_GLTEXTURES_HPP

#include "threepp/textures/Texture.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLProperties.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include <glad/glad.h>

#include <memory>
#include <unordered_map>

namespace threepp::gl {

    struct GLTextures {

        const GLint maxTextures = GLCapabilities::instance().maxTextures;
        const GLint maxCubemapSize = GLCapabilities::instance().maxCubemapSize;
        const GLint maxTextureSize = GLCapabilities::instance().maxTextureSize;
        const GLint maxSamples = GLCapabilities::instance().maxSamples;

        GLTextures(
                GLState &state,
                GLProperties &properties,
                GLInfo &info);

        void initTexture(GLTextureProperties::Properties &textureProperties, Texture &texture);

        void uploadTexture(GLTextureProperties::Properties &textureProperties, Texture &texture, GLint slot);

        void deallocateTexture( Texture &texture );

        void resetTextureUnits();

        int allocateTextureUnit();

    private:
        GLState &state;
        GLProperties &properties;
        GLInfo &info;

        EventListener onTextureDispose;

        int textureUnits = 0;
    };

}// namespace threepp::gl

#endif//THREEPP_GLTEXTURES_HPP
