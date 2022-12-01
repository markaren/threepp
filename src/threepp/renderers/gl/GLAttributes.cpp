
#include "threepp/renderers/gl/GLAttributes.hpp"

#include "threepp/utils/InstanceOf.hpp"

using namespace threepp;
using namespace threepp::gl;

Buffer GLAttributes::createBuffer(BufferAttribute *attribute, GLenum bufferType) {

    const auto usage = attribute->getUsage();

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(bufferType, buffer);

    GLint type;
    GLsizei bytesPerElement;
    if (instanceof <IntBufferAttribute>(attribute)) {
        type = GL_UNSIGNED_INT;
        bytesPerElement = sizeof(int);
        auto attr = dynamic_cast<IntBufferAttribute *>(attribute);
        const auto &array = attr->array();
        glBufferData(bufferType, (GLsizei) (array.size() * bytesPerElement), array.data(), usage);

    } else if (instanceof <FloatBufferAttribute>(attribute)) {
        type = GL_FLOAT;
        bytesPerElement = sizeof(float);
        auto attr = dynamic_cast<FloatBufferAttribute *>(attribute);
        const auto &array = attr->array();
        glBufferData(bufferType, (GLsizei) (array.size() * bytesPerElement), array.data(), usage);
    } else {

        throw std::runtime_error("TODO");
    }

    return {buffer, type, bytesPerElement, attribute->version + 1};
}

void GLAttributes::updateBuffer(GLuint buffer, BufferAttribute *attribute, GLenum bufferType, int bytesPerElement) {

    auto &updateRange = attribute->updateRange;

    glBindBuffer(bufferType, buffer);

    if (updateRange.count == -1) {

        if (instanceof <IntBufferAttribute>(attribute)) {

            auto attr = dynamic_cast<IntBufferAttribute *>(attribute);
            const auto &array = attr->array();
            glBufferSubData(bufferType, 0, (GLsizei) (array.size() * bytesPerElement), array.data());

        } else if (instanceof <FloatBufferAttribute>(attribute)) {

            auto attr = dynamic_cast<FloatBufferAttribute *>(attribute);
            const auto &array = attr->array();
            glBufferSubData(bufferType, 0, (GLsizei) (array.size() * bytesPerElement), array.data());
        } else {

            throw std::runtime_error("TODO");
        }

    } else {

        if (instanceof <IntBufferAttribute>(attribute)) {

            auto attr = dynamic_cast<IntBufferAttribute *>(attribute);
            const auto &array = attr->array();
            std::vector<int> sub(array.begin() + updateRange.offset, array.begin() + updateRange.offset + updateRange.count);
            glBufferSubData(bufferType, updateRange.offset * bytesPerElement, (GLsizei) (sub.size() * bytesPerElement), sub.data());

        } else if (instanceof <FloatBufferAttribute>(attribute)) {

            auto attr = dynamic_cast<FloatBufferAttribute *>(attribute);
            const auto &array = attr->array();
            std::vector<float> sub(array.begin() + updateRange.offset, array.begin() + updateRange.offset + updateRange.count);
            glBufferSubData(bufferType, updateRange.offset * bytesPerElement, (GLsizei) (sub.size() * bytesPerElement), sub.data());
        } else {

            throw std::runtime_error("TODO");
        }

        updateRange.count = -1;
    }
}

Buffer GLAttributes::get(BufferAttribute *attribute) {

    return buffers_.at(attribute);
}

void GLAttributes::remove(BufferAttribute *attribute) {

    if (buffers_.count(attribute)) {

        auto &data = buffers_.at(attribute);

        glDeleteBuffers(1, &data.buffer);

        buffers_.erase(attribute);
    }
}

void GLAttributes::update(BufferAttribute *attribute, GLenum bufferType) {

    if (!buffers_.count(attribute)) {

        buffers_[attribute] = createBuffer(attribute, bufferType);

    } else {

        auto &data = buffers_.at(attribute);

        if (data.version < attribute->version) {
            updateBuffer(data.buffer, attribute, bufferType, data.bytesPerElement);
            data.version++;
        }
    }
}
