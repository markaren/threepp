// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Path.js

#ifndef THREEPP_PATH_HPP
#define THREEPP_PATH_HPP

#include "threepp/extras/core/CurvePath.hpp"
#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/extras/curves/LineCurve.hpp"

namespace threepp {

    class Path: public CurvePath<Vector2> {

    public:
        explicit Path(const std::vector<Vector2>& points = {}) {

            setFromPoints(points);
        }

        Path& setFromPoints(const std::vector<Vector2>& points) {

            if (!points.empty()) {

                this->moveTo(points[0].x, points[0].y);

                for (unsigned i = 1, l = points.size(); i < l; i++) {

                    this->lineTo(points[i].x, points[i].y);
                }
            }

            return *this;
        }

        Path& moveTo(float x, float y) {

            this->currentPoint.set(x, y);// TODO consider referencing vectors instead of copying?

            return *this;
        }

        Path& lineTo(float x, float y) {

            auto curve = std::make_unique<LineCurve>(this->currentPoint, Vector2(x, y));
            this->curves.emplace_back(std::move(curve));

            this->currentPoint.set(x, y);

            return *this;
        }

        Path& absarc(float aX, float aY, float aRadius, float aStartAngle, float aEndAngle, bool aClockwise = false) {

            this->absellipse(aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise);

            return *this;
        }

        Path& absellipse(float aX, float aY, float xRadius, float yRadius, float aStartAngle, float aEndAngle, bool aClockwise = false, float aRotation = 0) {

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


    private:
        Vector2 currentPoint;
    };

}// namespace threepp

#endif//THREEPP_PATH_HPP
