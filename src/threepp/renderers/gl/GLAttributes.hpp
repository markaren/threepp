// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLAttributes.js

#ifndef THREEPP_GLATTRIBUTES_HPP
#define THREEPP_GLATTRIBUTES_HPP

#include "GLCapabilities.hpp"

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/utils/InstanceOf.hpp"

#include <glad/glad.h>

namespace threepp::gl {

    struct Buffer {
        GLuint buffer;
        GLint type;
        GLsizei bytesPerElement;
        unsigned int version;
    };

    struct GLAttributes {

        Buffer createBuffer(BufferAttribute *attribute, GLenum bufferType) {

            const auto usage = attribute->getUsage();

            GLuint buffer;
            glCreateBuffers(1, &buffer);
            glBindBuffer(bufferType, buffer);

            GLint type;
            GLsizei bytesPerElement;
            if (instanceof <IntBufferAttribute>(attribute)) {
                auto attr = dynamic_cast<IntBufferAttribute *>(attribute);
                auto array = attr->array();
                glBufferData(bufferType, (GLsizei) array.size(), array.data(), usage);
                type = GL_UNSIGNED_INT;
                bytesPerElement = sizeof(int);
            } else if (instanceof <FloatBufferAttribute>(attribute)) {
                auto attr = dynamic_cast<FloatBufferAttribute *>(attribute);
                auto array = attr->array();
                glBufferData(bufferType, (GLsizei) array.size(), array.data(), usage);
                type = GL_FLOAT;
                bytesPerElement = sizeof(float);
            }

            return {buffer, type, bytesPerElement, attribute->version()};
        }

        void updateBuffer(GLuint buffer, BufferAttribute *attribute, GLenum bufferType, int bytesPerElement) {

            auto &updateRange = attribute->getUpdateRange();

            glBindBuffer(bufferType, buffer);

            if (updateRange.count == -1) {

                if (instanceof <IntBufferAttribute>(attribute)) {

                    auto attr = dynamic_cast<IntBufferAttribute *>(attribute);
                    auto array = attr->array();
                    glBufferSubData(bufferType, 0, (GLsizei) array.size(), array.data());

                } else if (instanceof <FloatBufferAttribute>(attribute)) {

                    auto attr = dynamic_cast<FloatBufferAttribute *>(attribute);
                    auto array = attr->array();
                    glBufferSubData(bufferType, 0, (GLsizei) array.size(), array.data());
                }

            } else {

                if (dynamic_cast<IntBufferAttribute *>(attribute)) {

                    auto attr = dynamic_cast<IntBufferAttribute *>(attribute);
                    auto array = attr->array();
                    std::vector<int> sub(array.begin() + updateRange.offset, array.begin() + updateRange.offset + updateRange.count);
                    glBufferSubData(bufferType, updateRange.offset * bytesPerElement, (GLsizei) sub.size(), sub.data());

                } else if (dynamic_cast<FloatBufferAttribute *>(attribute)) {

                    auto attr = dynamic_cast<FloatBufferAttribute *>(attribute);
                    auto array = attr->array();
                    std::vector<float> sub(array.begin() + updateRange.offset, array.begin() + updateRange.offset + updateRange.count);
                    glBufferSubData(bufferType, updateRange.offset * bytesPerElement, (GLsizei) sub.size(), sub.data());
                }

                updateRange.count = -1;
            }
        }

        Buffer get(BufferAttribute *attribute) {

            return buffers_.at(attribute);
        }

        void remove(BufferAttribute *attribute) {

            if (buffers_.count(attribute)) {

                auto &data = buffers_.at(attribute);

                glDeleteBuffers(1, &data.buffer);
            }
        }

        void update(BufferAttribute *attribute, GLenum bufferType) {

            if (!buffers_.count(attribute)) {

                buffers_[attribute] = createBuffer(attribute, bufferType);

            } else {

                auto &data = buffers_.at(attribute);
                updateBuffer(data.buffer, attribute, bufferType, data.bytesPerElement);
                data.version++;
            }
        }

    private:
        std::unordered_map<BufferAttribute *, Buffer> buffers_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLATTRIBUTES_HPP
