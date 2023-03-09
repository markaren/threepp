// // https://github.com/mrdoob/three.js/blob/r129/src/geometries/ShapeGeometry.js

#ifndef THREEPP_SHAPEGEOMETRY_HPP
#define THREEPP_SHAPEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Shape.hpp"

namespace threepp {

    class ShapeGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<ShapeGeometry> create(Shape& shape, unsigned int curveSegments = 12) {

            return std::shared_ptr<ShapeGeometry>(new ShapeGeometry({&shape}, curveSegments));
        }

        static std::shared_ptr<ShapeGeometry> create(const std::vector<std::shared_ptr<Shape>>& shapes, unsigned int curveSegments = 12) {

            std::vector<Shape*> ptrs(shapes.size());
            std::transform(shapes.begin(), shapes.end(), ptrs.begin(), [&](auto& shape) { return shape.get(); });

            return std::shared_ptr<ShapeGeometry>(new ShapeGeometry(ptrs, curveSegments));
        }

        static std::shared_ptr<ShapeGeometry> create(const std::vector<Shape*>& shapes, unsigned int curveSegments = 12) {

            return std::shared_ptr<ShapeGeometry>(new ShapeGeometry(shapes, curveSegments));
        }

    protected:
        ShapeGeometry(const std::vector<Shape*>& shapes, unsigned int curveSegments);
    };

}// namespace threepp

#endif//THREEPP_SHAPEGEOMETRY_HPP
