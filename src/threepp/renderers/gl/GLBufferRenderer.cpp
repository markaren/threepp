
#pragma warning(disable : 4312)

#include "threepp/renderers/gl/GLBufferRenderer.hpp"

#ifndef EMSCRIPTEN
#include <glad/glad.h>
#else
#include <GLES3/gl3.h>
#endif

using namespace threepp;
using namespace threepp::gl;

void BufferRenderer::setMode(GLenum mode) {

    this->mode_ = mode;
}

void GLBufferRenderer::render(int start, int count) {

    glDrawArrays(mode_, start, count);

    info_.update(count, mode_, 1);
}

void GLBufferRenderer::renderInstances(int start, int count, int primcount) {

    if (primcount == 0) return;

    glDrawArraysInstanced(mode_, start, count, primcount);

    info_.update(count, mode_, primcount);
}

void GLIndexedBufferRenderer::setIndex(const Buffer& value) {

    type_ = value.type;
    bytesPerElement_ = value.bytesPerElement;
}

void GLIndexedBufferRenderer::render(int start, int count) {

    glDrawElements(mode_, count, type_, (GLvoid*) (start * bytesPerElement_));

    info_.update(count, mode_, 1);
}

void GLIndexedBufferRenderer::renderInstances(int start, int count, int primcount) {

    if (primcount == 0) return;

    glDrawElementsInstanced(mode_, count, type_, (GLvoid*) (start * bytesPerElement_), primcount);

    info_.update(count, mode_, primcount);
}
