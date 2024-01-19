
#include "threepp/renderers/gl/GLInfo.hpp"

#include <iostream>

#if EMSCRIPTEN
#include <GL/gl.h>
#else
#include <glad/glad.h>
#endif

using namespace threepp;


void gl::GLInfo::update(int count, unsigned int mode, size_t instanceCount) {

    ++render.calls;

    switch (mode) {

        case GL_TRIANGLES:
            render.triangles += instanceCount * (count / 3);
            break;

        case GL_LINES:
            render.lines += instanceCount * (count / 2);
            break;

        case GL_LINE_STRIP:
            render.lines += instanceCount * (count - 1);
            break;

        case GL_LINE_LOOP:
            render.lines += instanceCount * count;
            break;

        case GL_POINTS:
            render.points += instanceCount * count;
            break;

        default:
            std::cerr << "THREE.GLInfo: Unknown draw mode: " << mode << std::endl;
            break;
    }
}
void gl::GLInfo::reset() {

    ++render.frame;
    render.calls = 0;
    render.triangles = 0;
    render.points = 0;
    render.lines = 0;
}
