// https://github.com/mrdoob/three.js/blob/r129/src/extras/curves/CatmullRomCurve3.js

#ifndef THREEPP_CATMULLROMCURVE3_HPP
#define THREEPP_CATMULLROMCURVE3_HPP

#include <memory>

#include "threepp/extras/core/Curve.hpp"

namespace threepp {


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

        explicit CatmullRomCurve3(std::vector<Vector3> points = {}, bool closed = false, CurveType type = centripetal, float tension = 0.5f);

        void getPoint(float t, Vector3& point) const override;

        ~CatmullRomCurve3() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };


}// namespace threepp

#endif//THREEPP_CATMULLROMCURVE3_HPP
