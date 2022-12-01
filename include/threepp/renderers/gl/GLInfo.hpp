// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLInfo.js

#ifndef THREEPP_GLINFO_HPP
#define THREEPP_GLINFO_HPP

#include "GLProgram.hpp"

#include <memory>
#include <vector>

namespace threepp::gl {

    struct MemoryInfo {

        int geometries{0};
        int textures{0};
    };

    struct RenderInfo {

        int frame{0};
        int calls{0};
        int triangles{0};
        int points{0};
        int lines{0};
    };

    struct GLInfo {

        MemoryInfo memory{};
        RenderInfo render{};

        bool autoReset = true;

        std::vector<std::shared_ptr<GLProgram>> *programs;

        void update(int count, unsigned int mode, int instanceCount);

        void reset();
    };

}// namespace threepp::gl

#endif//THREEPP_GLINFO_HPP
