// https://github.com/mrdoob/three.js/blob/dev/src/geometries/BoxGeometry.js

#ifndef THREEPP_BOX_GEOMETRY_HPP
#define THREEPP_BOX_GEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <cmath>

namespace threepp {

    class BoxGeometry : public BufferGeometry {

    public:
        const float width;
        const float height;
        const float depth;
        const int widthSegments;
        const int heightSegments;
        const int depthSegments;

        static std::shared_ptr<BoxGeometry> create(float width = 1, float height = 1, float depth = 1, int widthSegments = 1, int heightSegments = 1, int depthSegments = 1) {
            return std::shared_ptr<BoxGeometry>(new BoxGeometry(width, height, depth, widthSegments, heightSegments, depthSegments));
        }

    protected:
        explicit BoxGeometry(float width, float height, float depth, int widthSegments, int heightSegments, int depthSegments);
    };

}// namespace threepp

#endif//THREEPP_BOX_GEOMETRY_HPP
