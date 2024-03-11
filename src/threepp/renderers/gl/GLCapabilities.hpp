// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLCapabilities.js

#ifndef THREEPP_GLCAPABILITIES_HPP
#define THREEPP_GLCAPABILITIES_HPP

#include "threepp/renderers/gl/GLUtils.hpp"

#include <ostream>
#include <string>

namespace threepp::gl {

    struct GLCapabilities {

        const int maxAnisotropy;

        const std::string precision = "highp";

        const bool drawBuffers = true;

        const bool logarithmicDepthBuffer = false;

        const int maxTextures;
        const int maxVertexTextures;
        const int maxTextureSize;
        const int maxCubemapSize;

        const int maxAttributes;
        const int maxVertexUniforms;
        const int maxVaryings;
        const int maxFragmentUniforms;

        const bool vertexTextures;
        const bool floatFragmentTextures;
        const bool floatVertexTextures;

        const int maxSamples;

        GLCapabilities(const GLCapabilities&) = delete;
        void operator=(const GLCapabilities&) = delete;

        friend std::ostream& operator<<(std::ostream& os, const GLCapabilities& v) {
            os << "GLCapabilities(\n"
               << " maxAnisotropy: " << v.maxAnisotropy << "\n"
               << " maxTextures: " << v.maxTextures << "\n"
               << " maxVertexTextures: " << v.maxVertexTextures << "\n"
               << " maxTextureSize: " << v.maxTextureSize << "\n"
               << " maxCubemapSize: " << v.maxCubemapSize << "\n"
               << " maxAttributes: " << v.maxAttributes << "\n"
               << " maxVertexUniforms: " << v.maxVertexUniforms << "\n"
               << " maxVaryings: " << v.maxVaryings << "\n"
               << " maxFragmentUniforms: " << v.maxFragmentUniforms << "\n"
               << " vertexTextures: " << (v.vertexTextures ? "true" : "false") << "\n"
               << " maxSamples: " << v.maxSamples << "\n"
               << ")";
            return os;
        }

        static GLCapabilities& instance() {
            static GLCapabilities instance;
            return instance;
        }

    private:
        GLCapabilities()
            : maxAnisotropy(static_cast<int>(glGetParameterf(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))),

              maxTextures(glGetParameteri(GL_MAX_TEXTURE_IMAGE_UNITS)),
              maxVertexTextures(glGetParameteri(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)),
              maxTextureSize(glGetParameteri(GL_MAX_TEXTURE_SIZE)),
              maxCubemapSize(glGetParameteri(GL_MAX_CUBE_MAP_TEXTURE_SIZE)),

              maxAttributes(glGetParameteri(GL_MAX_VERTEX_ATTRIBS)),
              maxVertexUniforms(glGetParameteri(GL_MAX_VERTEX_UNIFORM_VECTORS)),
              maxVaryings(glGetParameteri(GL_MAX_VARYING_VECTORS)),
              maxFragmentUniforms(glGetParameteri(GL_MAX_FRAGMENT_UNIFORM_VECTORS)),

              vertexTextures(maxVertexTextures > 0),
              floatFragmentTextures(GL_ARB_texture_float),
              floatVertexTextures(vertexTextures && floatFragmentTextures),

              maxSamples(glGetParameteri(GL_MAX_SAMPLES)) {}
    };

}// namespace threepp::gl

#endif//THREEPP_GLCAPABILITIES_HPP
