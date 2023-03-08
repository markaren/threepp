// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CatmullRomCurve3.js

#ifndef THREEPP_CATMULLROMCURVE3_HPP
#define THREEPP_CATMULLROMCURVE3_HPP

#include <utility>

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class CubicPoly {

    public:
        float c0{}, c1{}, c2{}, c3{};

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


    /**
     * Centripetal CatmullRom Curve - which is useful for avoiding
     * cusps and self-intersections in non-uniform catmull rom curves.
     * http://www.cemyuksel.com/research/catmullrom_param/catmullrom.pdf
     *
     * curve.type accepts centripetal(default), chordal and catmullrom
     * curve.tension is used for catmullrom which defaults to 0.5
     */
    class CatmullRomCurve3: public Curve3 {

    public:
        enum CurveType {
            centripetal,
            chordal,
            catmullrom
        };

        std::vector<Vector3> points;
        bool closed;
        CurveType curveType;
        float tension;

        void getPoint(float t, Vector3& point) override;

        static std::shared_ptr<CatmullRomCurve3> create(const std::vector<Vector3>& points = {}, bool closed = false, CurveType type = CurveType::centripetal, float tension = 0.5f) {

            return std::shared_ptr<CatmullRomCurve3>(new CatmullRomCurve3(points, closed, type, tension));
        }

    private:
        Vector3 tmp;
        CubicPoly px, py, pz;

        CatmullRomCurve3(std::vector<Vector3> points, bool closed, CurveType type, float tension)
            : points(std::move(points)), closed(closed), curveType(type), tension(tension) {}
    };


}// namespace threepp

#endif//THREEPP_CATMULLROMCURVE3_HPP
