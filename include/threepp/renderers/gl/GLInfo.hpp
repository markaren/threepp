// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLInfo.js

#ifndef THREEPP_GLINFO_HPP
#define THREEPP_GLINFO_HPP

#include "GLProgram.hpp"

#include <memory>
#include <vector>
#include <ostream>

namespace threepp::gl {

    struct MemoryInfo {

        int geometries{0};
        int textures{0};

        friend std::ostream& operator<<(std::ostream& os, const MemoryInfo& m) {
            os << "MemoryInfo: geomestries=" << m.geometries << ", textures=" << m.textures;
            return os;
        }
    };

    struct RenderInfo {

        int frame{0};
        int calls{0};
        int triangles{0};
        int points{0};
        int lines{0};

        friend std::ostream& operator<<(std::ostream& os, const RenderInfo& m) {
            os << "RenderInfo: frame=" << m.frame << ", calls=" << m.calls << ", triangles=" << m.triangles << ", points=" << m.points << ", lines=" << m.lines;
            return os;
        }
    };

    struct GLInfo {

        MemoryInfo memory{};
        RenderInfo render{};

        bool autoReset = true;

        void update(int count, unsigned int mode, int instanceCount);

        void reset();

        friend std::ostream& operator<<(std::ostream& os, const GLInfo& m) {
            os << m.memory << "\n" << m.render;
            return os;
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLINFO_HPP
