
#include "GLInfo.hpp"

#include <glad/glad.h>

#include <iostream>

void threepp::gl::GLInfo::update(int count, int mode, int instanceCount) {

    render.calls++;

    switch (mode) {

        case GL_TRIANGLES:
            render.triangles += instanceCount * (count / 3);
            break;

        case GL_LINES:
            render.lines += instanceCount * ( count / 2 );
            break;

        case GL_LINE_STRIP:
            render.lines += instanceCount * ( count - 1 );
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
void threepp::gl::GLInfo::reset() {

    render.frame ++;
    render.calls = 0;
    render.triangles = 0;
    render.points = 0;
    render.lines = 0;
}
