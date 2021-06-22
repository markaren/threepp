// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLCapabilities.js

#ifndef THREEPP_GLCAPABILITIES_HPP
#define THREEPP_GLCAPABILITIES_HPP

#include <glad/glad.h>

namespace threepp::gl {

    struct GLCapabilities {

        GLint64 maxAnisotropy;

        const std::string precision = "highp";

        bool logarithmicDepthBuffer = false;

        GLint64 maxTextures;
        GLint64 maxVertexTextures;
        GLint64 maxTextureSize;
        GLint64 maxCubemapSize;

        GLint64 maxAttributes;
        GLint64 maxVertexUniforms;
        GLint64 maxVaryings ;
        GLint64 maxFragmentUniforms ;

        bool vertexTextures;
        bool floatVertexTextures = true;

        GLint64 maxSamples;

        GLCapabilities() {

            glGetInteger64v(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy);

            glGetInteger64v(GL_MAX_TEXTURE_IMAGE_UNITS, (GLint64 *) &maxTextures);
            glGetInteger64v(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, (GLint64 *) &maxVertexTextures);
            glGetInteger64v(GL_MAX_TEXTURE_SIZE, (GLint64 *) &maxTextureSize);
            glGetInteger64v(GL_MAX_CUBE_MAP_TEXTURE_SIZE, (GLint64 *) &maxCubemapSize);

            glGetInteger64v(GL_MAX_VERTEX_ATTRIBS, (GLint64 *) &maxAttributes);
            glGetInteger64v(GL_MAX_VERTEX_UNIFORM_VECTORS, (GLint64 *) &maxVertexTextures);
            glGetInteger64v(GL_MAX_VARYING_VECTORS, (GLint64 *) &maxVaryings);
            glGetInteger64v(GL_MAX_FRAGMENT_UNIFORM_VECTORS, (GLint64 *) &maxFragmentUniforms);

            vertexTextures = maxVertexTextures > 0;

            glGetInteger64v(GL_MAX_SAMPLES, (GLint64 *) &maxSamples);
        }


    };

}// namespace threepp::gl

#endif//THREEPP_GLCAPABILITIES_HPP
