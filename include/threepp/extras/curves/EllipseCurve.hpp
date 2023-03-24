// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/EllipseCurve.js

#ifndef THREEPP_ELLIPSECURVE_HPP
#define THREEPP_ELLIPSECURVE_HPP

#include "threepp/extras/core/Curve.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class EllipseCurve: public Curve<Vector2> {

    public:
        explicit EllipseCurve(
                float aX = 0, float aY = 0,
                float xRadius = 1, float yRadius = 1,
                float aStartAngle = 0, float aEndAngle = math::TWO_PI,
                bool aClockwise = false, float aRotation = 0);

        void getPoint(float t, Vector2& target) const override;

    private:
        float aX;
        float aY;

        float xRadius;
        float yRadius;

        float aStartAngle;
        float aEndAngle;

        bool aClockwise;

        float aRotation;
    };

}// namespace threepp

#endif//THREEPP_ELLIPSECURVE_HPP
