
#include "threepp/extras/curves/CatmullRomCurve3.hpp"

#include <cmath>
#include <utility>

using namespace threepp;


struct CatmullRomCurve3::Impl {

    struct CubicPoly {

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

    Vector3 tmp;
    CubicPoly px, py, pz;
    CatmullRomCurve3& scope;

    explicit Impl(CatmullRomCurve3& scope): scope(scope) {}

    void getPoint(float t, Vector3& point) {

        const auto l = scope.points.size();

        const float p = (static_cast<float>(l) - (scope.closed ? 0.f : 1.f)) * t;
        size_t intPoint = std::floor(p);
        float weight = p - static_cast<float>(intPoint);

        if (scope.closed) {

            intPoint += intPoint > 0 ? 0 : static_cast<int>(std::floor(std::abs(static_cast<int>(intPoint)) / static_cast<float>(l)) + 1) * l;

        } else if (weight == 0 && intPoint == l - 1) {

            intPoint = l - 2;
            weight = 1;
        }

        Vector3 p0, p3;// 4 points (p1 & p2 defined below)

        if (scope.closed || intPoint > 0) {

            p0 = scope.points[(static_cast<int>(intPoint) - 1) % l];

        } else {

            // extrapolate first point
            tmp.subVectors(scope.points[0], scope.points[1]).add(scope.points[0]);
            p0 = tmp;
        }

        const auto p1 = scope.points[static_cast<int>(intPoint) % l];
        const auto p2 = scope.points[(static_cast<int>(intPoint) + 1) % l];

        if (scope.closed || intPoint + 2 < l) {

            p3 = scope.points[(static_cast<int>(intPoint) + 2) % l];

        } else {

            // extrapolate last point
            tmp.subVectors(scope.points[static_cast<int>(l) - 1], scope.points[static_cast<int>(l) - 2]).add(scope.points[static_cast<int>(l) - 1]);
            p3 = tmp;
        }

        if (scope.curveType == CurveType::centripetal || scope.curveType == CurveType::chordal) {

            // init Centripetal / Chordal Catmull-Rom
            const float pow = scope.curveType == CurveType::chordal ? 0.5f : 0.25f;
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

        } else if (scope.curveType == CurveType::catmullrom) {

            px.initCatmullRom(p0.x, p1.x, p2.x, p3.x, scope.tension);
            py.initCatmullRom(p0.y, p1.y, p2.y, p3.y, scope.tension);
            pz.initCatmullRom(p0.z, p1.z, p2.z, p3.z, scope.tension);
        }

        point.set(
                px.calc(weight),
                py.calc(weight),
                pz.calc(weight));
    }
};


CatmullRomCurve3::CatmullRomCurve3(std::vector<Vector3> points, bool closed, CatmullRomCurve3::CurveType type, float tension)
    : pimpl_(std::make_unique<Impl>(*this)), points(std::move(points)), closed(closed), curveType(type), tension(tension) {}


void CatmullRomCurve3::getPoint(float t, Vector3& point) const {

    pimpl_->getPoint(t, point);
}

threepp::CatmullRomCurve3::~CatmullRomCurve3() = default;
