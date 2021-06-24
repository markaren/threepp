
#include "threepp/renderers/GLRenderer.hpp"

#include <glad/glad.h>

using namespace threepp;

namespace {

    inline unsigned int createShader(int type, const char *str) {

        const auto shader = glCreateShader(type);

        glShaderSource(shader, 1, &str, nullptr);
        glCompileShader(shader);

        return shader;
    }

}

GLRenderer::GLRenderer(Canvas &canvas, const GLRenderer::Parameters &parameters)
    : canvas_(canvas), _width(canvas.getWidth()), _height(canvas.getHeight()),
      _viewport(0, 0, _width, _height),
      _scissor(0, 0, _width, _height), state(canvas),
      background(state, parameters.premultipliedAlpha) {
}

void GLRenderer::initGLContext() {
}

int GLRenderer::getTargetPixelRatio() const {
    return _pixelRatio;
}

void GLRenderer::getSize(Vector2 &target) const {
    target.set((float) _width, (float) _height);
}

void GLRenderer::setSize(int width, int height) {

    _width = width;
    _height = height;

    canvas_.setSize(width * _pixelRatio, height * _pixelRatio);
}

void GLRenderer::getDrawingBufferSize(Vector2 &target) const {

    target.set((float) (_width * _pixelRatio), (float) (_height * _pixelRatio)).floor();
}

void GLRenderer::setDrawingBufferSize(int width, int height, int pixelRatio) {

    _width = width;
    _height = height;

    _pixelRatio = pixelRatio;

    canvas_.setSize(width * pixelRatio, height * pixelRatio);

    this->setViewport(0, 0, width, height);
}

void GLRenderer::getCurrentViewport(Vector4 &target) const {

    target.copy(_currentViewport);
}

void GLRenderer::getViewport(Vector4 &target) const {

    target.copy(_viewport);
}

void GLRenderer::setViewport(const Vector4 &v) {

    _viewport.copy(v);

    state.viewport(_currentViewport.copy(_viewport).multiplyScalar((float) _pixelRatio).floor());
}

void GLRenderer::setViewport(int x, int y, int width, int height) {

    _viewport.set((float) x, (float) y, (float) width, (float) height);

    state.viewport(_currentViewport.copy(_viewport).multiplyScalar((float) _pixelRatio).floor());
}

void GLRenderer::getScissor(Vector4 &target) {

    target.copy(_scissor);
}

void GLRenderer::setScissor(const Vector4 &v) {

    _scissor.copy(v);

    state.scissor(_currentScissor.copy(_scissor).multiplyScalar((float) _pixelRatio).floor());
}


void GLRenderer::setScissor(int x, int y, int width, int height) {

    _scissor.set((float) x, (float) y, (float) width, (float) height);

    state.scissor(_currentScissor.copy(_scissor).multiplyScalar((float) _pixelRatio).floor());
}

bool GLRenderer::getScissorTest() const {

    return _scissorTest;
}

void GLRenderer::setScissorTest(bool boolean) {

    _scissorTest = boolean;

    state.setScissorTest(_scissorTest);
}
