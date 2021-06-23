// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLCapabilities.js

#ifndef THREEPP_GLCAPABILITIES_HPP
#define THREEPP_GLCAPABILITIES_HPP

#include <glad/glad.h>

#include <string>

namespace threepp::gl {

    struct GLCapabilities {

        GLint maxAnisotropy;

        const std::string precision = "highp";

        bool logarithmicDepthBuffer = false;

        GLint maxTextures;
        GLint maxVertexTextures;
        GLint maxTextureSize;
        GLint maxCubemapSize;

        GLint maxAttributes;
        GLint maxVertexUniforms;
        GLint maxVaryings ;
        GLint maxFragmentUniforms ;

        bool vertexTextures;
        bool floatVertexTextures = true;

        GLint64 maxSamples;

        GLCapabilities() {

            glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy);

            glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, (GLint *) &maxTextures);
            glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, (GLint *) &maxVertexTextures);
            glGetIntegerv(GL_MAX_TEXTURE_SIZE, (GLint *) &maxTextureSize);
            glGetIntegerv(GL_MAX_CUBE_MAP_TEXTURE_SIZE, (GLint *) &maxCubemapSize);

            glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, (GLint *) &maxAttributes);
            glGetIntegerv(GL_MAX_VERTEX_UNIFORM_VECTORS, (GLint *) &maxVertexTextures);
            glGetIntegerv(GL_MAX_VARYING_VECTORS, (GLint *) &maxVaryings);
            glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_VECTORS, (GLint *) &maxFragmentUniforms);

            vertexTextures = maxVertexTextures > 0;

            glGetInteger64v(GL_MAX_SAMPLES, (GLint64 *) &maxSamples);
        }


    };

}// namespace threepp::gl

#endif//THREEPP_GLCAPABILITIES_HPP
