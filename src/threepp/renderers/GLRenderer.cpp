
#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/constants.hpp"

using namespace threepp;

GLRenderer::GLRenderer(Canvas &canvas, const GLRenderer::Parameters &parameters)
    : canvas_(canvas), _width(canvas.getWidth()), _height(canvas.getHeight()), _viewport(0, 0, _width, _height), _scissor(0, 0, _width, _height), state(canvas) {
}

void GLRenderer::initGLContext() {
}

int GLRenderer::getTargetPixelRatio() const {
    return _pixelRatio;
}

void GLRenderer::getSize(Vector2 &target) const {
    target.set( (float) _width, (float) _height );
}

void GLRenderer::setSize(int width, int height) {

    _width = width;
    _height = height;

    canvas_.setSize(width * _pixelRatio, height * _pixelRatio);

}

void GLRenderer::getCurrentViewport(Vector4 &target) const {

    target.copy( _currentViewport );

}

void GLRenderer::getViewport(Vector4 &target) const {

    target.copy( _viewport );

}

void GLRenderer::setViewport(const Vector4 &v) {

    _viewport.copy(v);

    state.viewport( _currentViewport.copy( _viewport ).multiplyScalar( (float) _pixelRatio ).floor() );
}

void GLRenderer::setViewport(int x, int y, int width, int height) {

    _viewport.set( (float) x, (float) y, (float) width, (float) height );

    state.viewport( _currentViewport.copy( _viewport ).multiplyScalar( (float) _pixelRatio ).floor() );

}

void GLRenderer::getDrawingBufferSize(Vector2 &target) const {

    target.set( (float) (_width * _pixelRatio), (float)(_height * _pixelRatio) ).floor();
}
