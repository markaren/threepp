// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLCapabilities.js

#ifndef THREEPP_GLCAPABILITIES_HPP
#define THREEPP_GLCAPABILITIES_HPP

#include <iostream>
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
        const bool floatVertexTextures = true;

        const int maxSamples;

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
        GLCapabilities();
    };

}// namespace threepp::gl

#endif//THREEPP_GLCAPABILITIES_HPP
