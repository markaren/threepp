
#include "threepp/renderers/gl/GLAttributes.hpp"
#include "threepp/core/Assert.hpp"
#include "threepp/core/InterleavedBufferAttribute.hpp"

#include <cstddef>
#include <stdexcept>

#ifndef __EMSCRIPTEN__
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

using namespace threepp;
using namespace threepp::gl;

namespace {

    // GLBindingStates already dispatches glVertexAttribIPointer vs
    // glVertexAttribPointer off this enum and forwards the attribute's
    // `normalized` flag, so widening the table here is all that a narrow
    // attribute type needs to render correctly.
    GLint glTypeOf(AttributeType type) {

        switch (type) {
            case AttributeType::Float: return GL_FLOAT;
            case AttributeType::UInt32: return GL_UNSIGNED_INT;
            case AttributeType::UInt16: return GL_UNSIGNED_SHORT;
            case AttributeType::Int16: return GL_SHORT;
            case AttributeType::UInt8: return GL_UNSIGNED_BYTE;
            case AttributeType::Int8: return GL_BYTE;
        }

        throw std::runtime_error("[GLAttributes] unhandled AttributeType");
    }

}// namespace

Buffer GLAttributes::createBuffer(BufferAttribute* attribute, GLenum bufferType) {

    const auto usage = attribute->getUsage();
    const auto type = attribute->type();

    GLuint buffer;
    glGenBuffers(1, &buffer);
    glBindBuffer(bufferType, buffer);

    glBufferData(bufferType,
                 static_cast<GLsizeiptr>(attribute->byteLength()),
                 attribute->data(),
                 as_integer(usage));

    return {buffer,
            glTypeOf(type),
            static_cast<int>(threepp::bytesPerElement(type)),
            attribute->version};// attribute->version + 1 (?)
}

void GLAttributes::updateBuffer(GLuint buffer, BufferAttribute* attribute, GLenum bufferType, int bytesPerElement) {

    auto& updateRange = attribute->updateRange;

    glBindBuffer(bufferType, buffer);

    // updateRange is expressed in scalar elements, not items.
    const auto* base = static_cast<const std::byte*>(attribute->data());

    if (updateRange.count == -1) {

        glBufferSubData(bufferType, 0, static_cast<GLsizeiptr>(attribute->byteLength()), base);

    } else {

        // Point glBufferSubData straight into the attribute's storage. This used
        // to materialise a fresh std::vector holding a copy of the sub-range on
        // every partial update — an allocation plus a memcpy per dirty attribute
        // per frame, for data the driver was about to copy again anyway.
        const auto offsetBytes = static_cast<GLintptr>(updateRange.offset) * bytesPerElement;
        const auto sizeBytes = static_cast<GLsizeiptr>(updateRange.count) * bytesPerElement;

        THREEPP_ASSERT_MSG(static_cast<size_t>(offsetBytes + sizeBytes) <= attribute->byteLength(),
                           "GLAttributes: updateRange extends past the attribute array");

        glBufferSubData(bufferType, offsetBytes, sizeBytes, base + offsetBytes);

        updateRange.count = -1;
    }
}

Buffer GLAttributes::get(BufferAttribute* attribute) const {

    if (auto attr = dynamic_cast<InterleavedBufferAttribute*>(attribute)) {
        attribute = attr->data.get();
    }

    return buffers_.at(attribute);
}

void GLAttributes::remove(BufferAttribute* attribute) {

    if (auto attr = dynamic_cast<InterleavedBufferAttribute*>(attribute)) {
        attribute = attr->data.get();
    }

    if (buffers_.contains(attribute)) {

        auto& data = buffers_.at(attribute);

        glDeleteBuffers(1, &data.buffer);

        buffers_.erase(attribute);
    }
}

void GLAttributes::update(BufferAttribute* attribute, GLenum bufferType) {

    if (auto attr = dynamic_cast<InterleavedBufferAttribute*>(attribute)) {
        attribute = attr->data.get();
    }

    if (!buffers_.contains(attribute)) {

        buffers_[attribute] = createBuffer(attribute, bufferType);

    } else {

        auto& data = buffers_.at(attribute);

        if (data.version < attribute->version) {
            updateBuffer(data.buffer, attribute, bufferType, data.bytesPerElement);
            ++data.version;
        }
    }
}
