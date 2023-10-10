
#include "threepp/extras/core/Path.hpp"

#include "threepp/extras/curves/CubicBezierCurve.hpp"
#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/extras/curves/LineCurve.hpp"
#include "threepp/extras/curves/QuadraticBezierCurve.hpp"
#include "threepp/extras/curves/SplineCurve.hpp"

using namespace threepp;


Path::Path(const std::vector<Vector2>& points) {

    setFromPoints(points);
}

Path& Path::setFromPoints(const std::vector<Vector2>& points) {

    if (!points.empty()) {

        this->moveTo(points[0].x, points[0].y);

        for (unsigned i = 1, l = points.size(); i < l; i++) {

            this->lineTo(points[i].x, points[i].y);
        }
    }

    return *this;
}

Path& Path::moveTo(float x, float y) {

    this->currentPoint.set(x, y);// TODO consider referencing vectors instead of copying?

    return *this;
}

Path& Path::lineTo(float x, float y) {

    auto curve = std::make_unique<LineCurve>(this->currentPoint, Vector2(x, y));
    this->curves.emplace_back(std::move(curve));

    this->currentPoint.set(x, y);

    return *this;
}

Path& Path::quadraticCurveTo(float aCPx, float aCPy, float aX, float aY) {

    auto curve = std::make_unique<QuadraticBezierCurve>(
            this->currentPoint.clone(),
            Vector2(aCPx, aCPy),
            Vector2(aX, aY));

    this->curves.emplace_back(std::move(curve));

    this->currentPoint.set(aX, aY);

    return *this;
}

Path& Path::bezierCurveTo(float aCP1x, float aCP1y, float aCP2x, float aCP2y, float aX, float aY) {

    auto curve = std::make_unique<CubicBezierCurve>(
            this->currentPoint.clone(),
            Vector2(aCP1x, aCP1y),
            Vector2(aCP2x, aCP2y),
            Vector2(aX, aY));

    this->curves.emplace_back(std::move(curve));

    this->currentPoint.set(aX, aY);

    return *this;
}

Path& Path::splineThru(const std::vector<Vector2>& pts) {

    auto npts = std::vector<Vector2>{this->currentPoint};
    npts.insert(npts.end(), pts.begin(), pts.end());

    auto curve = std::make_unique<SplineCurve>(npts);
    this->curves.emplace_back(std::move(curve));

    this->currentPoint.copy(pts[pts.size() - 1]);

    return *this;
}

Path& Path::arc(float aX, float aY, float aRadius, float aStartAngle, float aEndAngle, bool aClockwise) {

    const auto x0 = this->currentPoint.x;
    const auto y0 = this->currentPoint.y;

    this->absarc(aX + x0, aY + y0, aRadius,
                 aStartAngle, aEndAngle, aClockwise);

    return *this;
}

Path& Path::absarc(float aX, float aY, float aRadius, float aStartAngle, float aEndAngle, bool aClockwise) {

    this->absellipse(aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise);

    return *this;
}

Path& Path::absellipse(float aX, float aY, float xRadius, float yRadius, float aStartAngle, float aEndAngle, bool aClockwise, float aRotation) {

    auto curve = std::make_unique<EllipseCurve>(aX, aY, xRadius, yRadius, aStartAngle, aEndAngle, aClockwise, aRotation);

    if (!this->curves.empty()) {

        // if a previous curve is present, attempt to join
        Vector2 firstPoint;
        curve->getPoint(0, firstPoint);

        if (!firstPoint.equals(this->currentPoint)) {

            this->lineTo(firstPoint.x, firstPoint.y);
        }
    }

    this->curves.emplace_back(std::move(curve));

    Vector2 lastPoint;
    this->curves.back()->getPoint(1, lastPoint);
    this->currentPoint.copy(lastPoint);

    return *this;
}
