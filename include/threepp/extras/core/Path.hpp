// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Path.js

#ifndef THREEPP_PATH_HPP
#define THREEPP_PATH_HPP

#include "threepp/extras/core/CurvePath.hpp"

#include <optional>

namespace threepp {

    class Path: public CurvePath<Vector2> {

    public:
        explicit Path(const std::optional<std::vector<Vector2>>& points = {});

        Path& setFromPoints(const std::vector<Vector2>& points);

        Path& moveTo(float x, float y);

        Path& lineTo(float x, float y);

        Path& quadraticCurveTo(float aCPx, float aCPy, float aX, float aY);

        Path& bezierCurveTo(float aCP1x, float aCP1y, float aCP2x, float aCP2y, float aX, float aY);

        Path& splineThru(const std::vector<Vector2>& pts);

        Path& arc(float aX, float aY, float aRadius, float aStartAngle, float aEndAngle, bool aClockwise);

        Path& absarc(float aX, float aY, float aRadius, float aStartAngle, float aEndAngle, bool aClockwise = false);

        Path& absellipse(float aX, float aY, float xRadius, float yRadius, float aStartAngle, float aEndAngle, bool aClockwise = false, float aRotation = 0);


    private:
        Vector2 currentPoint;
    };

}// namespace threepp

#endif//THREEPP_PATH_HPP
