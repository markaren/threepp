// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/ShapePath.js

#ifndef THREEPP_SHAPEPATH_HPP
#define THREEPP_SHAPEPATH_HPP

#include "threepp/extras/core/Path.hpp"
#include "threepp/extras/core/Shape.hpp"
#include "threepp/math/Color.hpp"

#include <optional>
#include <vector>

namespace threepp {

    class ShapePath {

    public:
        Color color;
        Path* currentPath;
        std::vector<std::shared_ptr<Path>> subPaths;

        ShapePath& moveTo(float x, float y);

        ShapePath& lineTo(float x, float y);

        ShapePath& quadraticCurveTo(float aCPx, float aCPy, float aX, float aY);

        ShapePath& bezierCurveTo(float aCP1x, float aCP1y, float aCP2x, float aCP2y, float aX, float aY);

        ShapePath& splineThru(const std::vector<Vector2>& pts);

        [[nodiscard]] std::vector<Shape> toShapes(bool isCCW = false) const;
    };

}// namespace threepp

#endif//THREEPP_SHAPEPATH_HPP
