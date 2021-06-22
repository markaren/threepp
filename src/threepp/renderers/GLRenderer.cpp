
#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/constants.hpp"

using namespace threepp;

GLRenderer::GLRenderer(const Canvas &canvas, const GLRenderer::Parameters &parameters)
    : _width(canvas.getWidth()), _height(canvas.getHeight()), _viewPort(0, 0, _width, _height), _scissor(0, 0, _width, _height), state(canvas) {
}

void GLRenderer::initGLContext() {
}
