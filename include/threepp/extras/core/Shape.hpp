// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Shape.js

#ifndef THREEPP_SHAPE_HPP
#define THREEPP_SHAPE_HPP

#include "threepp/extras/core/Path.hpp"

#include <vector>

namespace threepp {

    struct ShapePoints {

        std::vector<Vector2> shape;
        std::vector<std::vector<Vector2>> holes;
    };

    class Shape: public Path {

    public:
        std::string uuid;

        std::vector<std::shared_ptr<Path>> holes;

        explicit Shape(const std::optional<std::vector<Vector2>>& points = {});

        std::vector<std::vector<Vector2>> getPointsHoles(unsigned int divisions) const;

        // get points of shape and holes (keypoints based on segments parameter)
        [[nodiscard]] ShapePoints extractPoints(unsigned int divisions) const;
    };

}// namespace threepp

#endif//THREEPP_SHAPE_HPP
