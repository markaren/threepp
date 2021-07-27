// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CatmullRomCurve3.js

#ifndef THREEPP_CATMULLROMCURVE3_HPP
#define THREEPP_CATMULLROMCURVE3_HPP

#include <utility>

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class CubicPoly;

    /**
     * Centripetal CatmullRom Curve - which is useful for avoiding
     * cusps and self-intersections in non-uniform catmull rom curves.
     * http://www.cemyuksel.com/research/catmullrom_param/catmullrom.pdf
     *
     * curve.type accepts centripetal(default), chordal and catmullrom
     * curve.tension is used for catmullrom which defaults to 0.5
     */
    class CatmullRomCurve3 : public Curve3 {

    public:
        enum CurveType {
            centripetal,
            chordal,
            catmullrom
        };

        std::vector<Vector3> points;
        bool closed;
        CurveType type;
        float tension;

        void getPoint(float t, Vector3 &point) {

            const auto l = static_cast<float>(points.size());

            const float p = (l - (this->closed ? 0.f : 1.f)) * t;
            float intPoint = std::floor(p);
            float weight = p - intPoint;

            if (this->closed) {

                intPoint += intPoint > 0 ? 0 : (std::floor(std::abs(intPoint) / l) + 1) * l;

            } else if (weight == 0 && intPoint == l - 1) {

                intPoint = l - 2;
                weight = 1;
            }

            float p0, p3;// 4 points (p1 & p2 defined below)

            if (this->closed || intPoint > 0) {

                p0 = points[(static_cast<int>(intPoint) - 1) % l];

            } else {

                // extrapolate first point
                tmp.subVectors(points[0], points[1]).add(points[0]);
                p0 = tmp;
            }

            const p1 = points[static_cast<int>(intPoint) % l];
            const p2 = points[(static_cast<int>(intPoint) + 1) % l];

            if (this->closed || intPoint + 2 < l) {

                p3 = points[(static_cast<int>(intPoint) + 2) % l];

            } else {

                // extrapolate last point
                tmp.subVectors(points[static_cast<int>(l) - 1], points[static_cast<int>(l) - 2]).add(points[static_cast<int>(l) - 1]);
                p3 = tmp;
            }

            if (this->curveType == CurveType::centripetal || this->curveType == CurveType::chordal) {

                // init Centripetal / Chordal Catmull-Rom
                const float pow = this->curveType == CurveType::chordal ? 0.5f : 0.25f;
                float dt0 = std::pow(p0.distanceToSquared(p1), pow);
                float dt1 = std::pow(p1.distanceToSquared(p2), pow);
                float dt2 = std::pow(p2.distanceToSquared(p3), pow);

                // safety check for repeated points
                if (dt1 < 1e-4) dt1 = 1.0;
                if (dt0 < 1e-4) dt0 = dt1;
                if (dt2 < 1e-4) dt2 = dt1;

                px.initNonuniformCatmullRom(p0.x, p1.x, p2.x, p3.x, dt0, dt1, dt2);
                py.initNonuniformCatmullRom(p0.y, p1.y, p2.y, p3.y, dt0, dt1, dt2);
                pz.initNonuniformCatmullRom(p0.z, p1.z, p2.z, p3.z, dt0, dt1, dt2);

            } else if (this->curveType == CurveType::catmullrom) {

                px.initCatmullRom(p0.x, p1.x, p2.x, p3.x, this->tension);
                py.initCatmullRom(p0.y, p1.y, p2.y, p3.y, this->tension);
                pz.initCatmullRom(p0.z, p1.z, p2.z, p3.z, this->tension);
            }

            point.set(
                    px.calc(weight),
                    py.calc(weight),
                    pz.calc(weight));
        }

        static std::shared_ptr<CatmullRomCurve3> create(const std::vector<Vector3>& points = {}, bool closed = false, CurveType type = CurveType::centripetal, float tension = 0.5f) {

            return std::shared_ptr<CatmullRomCurve3>(points, closed, type, tension);
        }

    private:
        Vector3 tmp;
        CubicPoly px{}, py{}, pz{};

        explicit CatmullRomCurve3(std::vector<Vector3> points, bool closed, CurveType type, float tension)
                : points(std::move(points)), closed(closed), type(type), tension(tension) {}

    };

    class CubicPoly {

    public:
        float c0, c1, c2, c3;

        void init(float x0, float x1, float t0, float t1) {

            c0 = x0;
            c1 = t0;
            c2 = -3 * x0 + 3 * x1 - 2 * t0 - t1;
            c3 = 2 * x0 - 2 * x1 + t0 + t1;
        }

        void initCatmullRom(float x0, float x1, float x2, float x3, float tension) {

            init(x1, x2, tension * (x2 - x0), tension * (x3 - x1));
        }

        void initNonuniformCatmullRom(float x0, float x1, float x2, float x3, float dt0, float dt1, float dt2) {

            // compute tangents when parameterized in [t1,t2]
            float t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
            float t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;

            // rescale tangents for parametrization in [0,1]
            t1 *= dt1;
            t2 *= dt1;

            init(x1, x2, t1, t2);
        }

        [[nodiscard]] float calc(float t) const {

            const float t2 = t * t;
            const float t3 = t2 * t;
            return c0 + c1 * t + c2 * t2 + c3 * t3;
        }
    };

}// namespace threepp

#endif//THREEPP_CATMULLROMCURVE3_HPP
