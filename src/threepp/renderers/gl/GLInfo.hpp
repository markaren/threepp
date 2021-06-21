// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLInfo.js

#ifndef THREEPP_GLINFO_HPP
#define THREEPP_GLINFO_HPP

#include <glad/glad.h>

#include "threepp/renderers/gl/GLProgram.hpp"

#include <iostream>
#include <vector>

namespace threepp::gl {

    struct MemoryInfo {

        int geometries;
        int textures;

    };

    struct RenderInfo {

        int frame;
        int calls;
        int triangles;
        int points;
        int lines;

    };

    class GLInfo {

    public:

        MemoryInfo memory;
        RenderInfo render;

        bool autoReset = true;

        std::vector<GLProgram> programs;

        void update(int count, int mode, int instanceCount) {

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

        void reset () {

            render.frame ++;
            render.calls = 0;
            render.triangles = 0;
            render.points = 0;
            render.lines = 0;
        }

    };

}

#endif//THREEPP_GLINFO_HPP
