// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLAttributes.js

#ifndef THREEPP_GLATTRIBUTES_HPP
#define THREEPP_GLATTRIBUTES_HPP

#include "GLCapabilities.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/utils/InstanceOf.hpp"

#include <glad/glad.h>

namespace threepp::gl {

    struct Buffer {
        GLuint buffer{};
        GLint type{};
        GLsizei bytesPerElement{};
        unsigned int version{};
    };

    struct GLAttributes {

        Buffer createBuffer(BufferAttribute *attribute, GLenum bufferType);

        void updateBuffer(GLuint buffer, BufferAttribute *attribute, GLenum bufferType, int bytesPerElement);

        Buffer get(BufferAttribute *attribute);

        void remove(BufferAttribute *attribute);

        void update(BufferAttribute *attribute, GLenum bufferType);

    private:
        std::unordered_map<BufferAttribute *, Buffer> buffers_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLATTRIBUTES_HPP
