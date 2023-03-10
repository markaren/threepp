// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CatmullRomCurve3.js

#ifndef THREEPP_CATMULLROMCURVE3_HPP
#define THREEPP_CATMULLROMCURVE3_HPP

#include <memory>
#include <utility>

#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class CubicPoly {

    public:
        float c0{}, c1{}, c2{}, c3{};

        void init(float x0, float x1, float t0, float t1);

        void initCatmullRom(float x0, float x1, float x2, float x3, float tension);

        void initNonuniformCatmullRom(float x0, float x1, float x2, float x3, float dt0, float dt1, float dt2);

        [[nodiscard]] float calc(float t) const;
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

        explicit CatmullRomCurve3(std::vector<Vector3> points = {}, bool closed = false, CurveType type = centripetal, float tension = 0.5f)
            : points(std::move(points)), closed(closed), curveType(type), tension(tension) {}

        void getPoint(float t, Vector3& point) override;

    private:
        Vector3 tmp;
        CubicPoly px, py, pz;
    };


}// namespace threepp

#endif//THREEPP_CATMULLROMCURVE3_HPP
