// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/geometries/ConvexGeometry.js

#ifndef THREEPP_CONVEXGEOMETRY_HPP
#define THREEPP_CONVEXGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class ConvexGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<ConvexGeometry> create(const std::vector<Vector3>& points);

    private:
        explicit ConvexGeometry(const std::vector<Vector3>& points);
    };

}// namespace threepp

#endif//THREEPP_CONVEXGEOMETRY_HPP
