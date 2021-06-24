// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLTextures.js

#ifndef THREEPP_GLTEXTURES_HPP
#define THREEPP_GLTEXTURES_HPP

#include "threepp/textures/Texture.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include <glad/glad.h>

#include <memory>
#include <unordered_map>

namespace threepp::gl {

    class GLTextures {

        struct TextureProperties {

            GLint glTexture;

        };

    public:
        GLTextures(
                std::shared_ptr<GLState> state,
                GLCapabilities capabilities,
                std::shared_ptr<GLInfo> info) {}

        void uploadTexture(const TextureProperties &textureProperties, GLint texture, GLint slot);

    };

}// namespace threepp::gl

#endif//THREEPP_GLTEXTURES_HPP
