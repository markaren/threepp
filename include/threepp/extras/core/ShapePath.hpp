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

        ShapePath& moveTo(float x, float y);

        ShapePath& lineTo(float x, float y);

        ShapePath& quadraticCurveTo(float aCPx, float aCPy, float aX, float aY);

        ShapePath& bezierCurveTo(float aCP1x, float aCP1y, float aCP2x, float aCP2y, float aX, float aY);

        ShapePath& splineThru(const std::vector<Vector2>& pts);

        [[nodiscard]] std::vector<std::shared_ptr<Shape>> toShapes(bool isCCW = false, bool noHoles = false) const;

    private:
        std::shared_ptr<Path> currentPath;
        std::vector<std::shared_ptr<Path>> subPaths;
    };

}// namespace threepp

#endif//THREEPP_SHAPEPATH_HPP
