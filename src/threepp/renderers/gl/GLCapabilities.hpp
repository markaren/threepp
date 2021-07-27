// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLCapabilities.js

#ifndef THREEPP_GLCAPABILITIES_HPP
#define THREEPP_GLCAPABILITIES_HPP

#include "glHelper.hpp"

#include <string>
#include <iostream>

namespace threepp::gl {

    struct GLCapabilities {

        const GLint maxAnisotropy;

        const std::string precision = "highp";

        const bool drawBuffers = true;

        const bool logarithmicDepthBuffer = false;

        const GLint maxTextures;
        const GLint maxVertexTextures;
        const GLint maxTextureSize;
        const GLint maxCubemapSize;

        const GLint maxAttributes;
        const GLint maxVertexUniforms;
        const GLint maxVaryings;
        const GLint maxFragmentUniforms;

        const bool vertexTextures;
        const bool floatVertexTextures = true;

        const GLint maxSamples;

        GLCapabilities(const GLCapabilities &) = delete;
        void operator=(const GLCapabilities &) = delete;

        friend std::ostream &operator<<(std::ostream &os, const GLCapabilities &v) {
            os << "GLCapabilities(\n"
                    << "maxAnisotropy: " << v.maxAnisotropy << "\n"
                    << "maxTextures: " << v.maxTextures << "\n"
                    << "maxVertexTextures: " << v.maxVertexTextures << "\n"
                    << "maxTextureSize: " << v.maxTextureSize << "\n"
                    << "maxCubemapSize: " << v.maxCubemapSize << "\n"
                    << "maxAttributes: " << v.maxAttributes << "\n"
                    << "maxVertexUniforms: " << v.maxVertexUniforms << "\n"
                    << "maxVaryings: " << v.maxVaryings << "\n"
                    << "maxFragmentUniforms: " << v.maxFragmentUniforms << "\n"
                    << "vertexTextures: " << (v.vertexTextures ? "true" : "false") << "\n"
                    << "maxSamples: " << v.maxSamples << "\n";
            return os;
        }

        static GLCapabilities &instance() {
            static GLCapabilities instance;
            return instance;
        }

    private:
        GLCapabilities()
            : maxAnisotropy(0),

              maxTextures(glGetParameter(GL_MAX_TEXTURE_IMAGE_UNITS)),
              maxVertexTextures(glGetParameter(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)),
              maxTextureSize(glGetParameter(GL_MAX_TEXTURE_SIZE)),
              maxCubemapSize(glGetParameter(GL_MAX_CUBE_MAP_TEXTURE_SIZE)),

              maxAttributes(glGetParameter(GL_MAX_VERTEX_ATTRIBS)),
              maxVertexUniforms(glGetParameter(GL_MAX_VERTEX_UNIFORM_VECTORS)),
              maxVaryings(glGetParameter(GL_MAX_VARYING_VECTORS)),
              maxFragmentUniforms(glGetParameter(GL_MAX_FRAGMENT_UNIFORM_VECTORS)),

              vertexTextures(maxVertexTextures > 0),

              maxSamples(glGetParameter(GL_MAX_SAMPLES)) {}


    };

}// namespace threepp::gl

#endif//THREEPP_GLCAPABILITIES_HPP
