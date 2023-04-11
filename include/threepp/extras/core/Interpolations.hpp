// https://github.com/mrdoob/three.js/blob/r129/src/extras/core/Interpolations.js

#ifndef THREEPP_INTERPOLATIONS_HPP
#define THREEPP_INTERPOLATIONS_HPP

namespace threepp::interpolants {

    /**
     * Bezier Curves formulas obtained from
     * http://en.wikipedia.org/wiki/Bézier_curve
     */

    inline float CatmullRom(float t, float p0, float p1, float p2, float p3) {

        const auto v0 = (p2 - p0) * 0.5f;
        const auto v1 = (p3 - p1) * 0.5f;
        const auto t2 = t * t;
        const auto t3 = t * t2;
        return (2 * p1 - 2 * p2 + v0 + v1) * t3 + (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 + v0 * t + p1;
    }

    //

    inline float QuadraticBezierP0(float t, float p) {

        const auto k = 1 - t;
        return k * k * p;
    }

    inline float QuadraticBezierP1(float t, float p) {

        return 2 * (1 - t) * t * p;
    }

    inline float QuadraticBezierP2(float t, float p) {

        return t * t * p;
    }

    inline float QuadraticBezier(float t, float p0, float p1, float p2) {

        return QuadraticBezierP0(t, p0) + QuadraticBezierP1(t, p1) +
               QuadraticBezierP2(t, p2);
    }

    //

    inline float CubicBezierP0(float t, float p) {

        const auto k = 1 - t;
        return k * k * k * p;
    }

    inline float CubicBezierP1(float t, float p) {

        const auto k = 1 - t;
        return 3 * k * k * t * p;
    }

    inline float CubicBezierP2(float t, float p) {

        return 3 * (1 - t) * t * t * p;
    }

    inline float CubicBezierP3(float t, float p) {

        return t * t * t * p;
    }

    inline float CubicBezier(float t, float p0, float p1, float p2, float p3) {

        return CubicBezierP0(t, p0) + CubicBezierP1(t, p1) + CubicBezierP2(t, p2) +
               CubicBezierP3(t, p3);
    }

}// namespace threepp::interpolants

#endif//THREEPP_INTERPOLATIONS_HPP
