// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLAttributes.js

#ifndef THREEPP_GLATTRIBUTES_HPP
#define THREEPP_GLATTRIBUTES_HPP

#include "GLCapabilities.hpp"

#include "threepp/core/BufferAttribute.hpp"

namespace threepp::gl {

    struct Buffer {
        unsigned int buffer{};
        int type{};
        int bytesPerElement{};
        unsigned int version{};
    };

    struct GLAttributes {

        Buffer createBuffer(BufferAttribute *attribute, unsigned int bufferType);

        void updateBuffer(unsigned int buffer, BufferAttribute *attribute, unsigned int bufferType, int bytesPerElement);

        Buffer get(BufferAttribute *attribute);

        void remove(BufferAttribute *attribute);

        void update(BufferAttribute *attribute, unsigned int bufferType);

    private:
        std::unordered_map<BufferAttribute *, Buffer> buffers_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLATTRIBUTES_HPP
