// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLInfo.js

#ifndef THREEPP_GLINFO_HPP
#define THREEPP_GLINFO_HPP

#include <memory>
#include <ostream>
#include <vector>

namespace threepp::gl {

    struct MemoryInfo {

        size_t geometries{0};
        size_t textures{0};

        friend std::ostream& operator<<(std::ostream& os, const MemoryInfo& m) {
            os << "MemoryInfo: geomestries=" << m.geometries << ", textures=" << m.textures;
            return os;
        }
    };

    struct RenderInfo {

        size_t frame{0};
        size_t calls{0};
        size_t triangles{0};
        size_t points{0};
        size_t lines{0};

        friend std::ostream& operator<<(std::ostream& os, const RenderInfo& m) {
            os << "RenderInfo: frame=" << m.frame << ", calls=" << m.calls << ", triangles=" << m.triangles << ", points=" << m.points << ", lines=" << m.lines;
            return os;
        }
    };

    struct GLInfo {

        MemoryInfo memory{};
        RenderInfo render{};

        bool autoReset = true;

        void update(int count, unsigned int mode, size_t instanceCount);

        void reset();

        friend std::ostream& operator<<(std::ostream& os, const GLInfo& m) {
            os << m.memory << "\n"
               << m.render;
            return os;
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLINFO_HPP
