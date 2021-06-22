// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLAttributes.js

#ifndef THREEPP_GLATTRIBUTES_HPP
#define THREEPP_GLATTRIBUTES_HPP

#include <glad/glad.h>

#include "threepp/core/BufferAttribute.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"

namespace threepp::gl {

    struct GLAttributes {

        template<class T>
        void createBuffer(BufferAttribute<T> &attribute, int bufferType) {

            const auto array = attribute.array();
            const auto usage = attribute.getUsage();

            GLuint buffer;
            glCreateBuffers(1, &buffer);

            glBindBuffer( bufferType, buffer );
            glBufferData( bufferType, array.data(), usage );

//            attribute.onUploadCallback();

            // TODO

        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLATTRIBUTES_HPP
