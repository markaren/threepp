
#include "threepp/extras/curve/CatmullRomCurve3.hpp"

using namespace threepp;


void CatmullRomCurve3::getPoint(float t, Vector3& point) {

    const auto l = points.size();

    const float p = (static_cast<float>(l) - (this->closed ? 0.f : 1.f)) * t;
    size_t intPoint = std::floor(p);
    float weight = p - static_cast<float>(intPoint);

    if (this->closed) {

        intPoint += intPoint > 0 ? 0 : static_cast<int>(std::floor(std::abs(static_cast<int>(intPoint)) / static_cast<float>(l)) + 1) * l;

    } else if (weight == 0 && intPoint == l - 1) {

        intPoint = l - 2;
        weight = 1;
    }

    Vector3 p0, p3;// 4 points (p1 & p2 defined below)

    if (this->closed || intPoint > 0) {

        p0 = points[(static_cast<int>(intPoint) - 1) % l];

    } else {

        // extrapolate first point
        tmp.subVectors(points[0], points[1]).add(points[0]);
        p0 = tmp;
    }

    const auto p1 = points[static_cast<int>(intPoint) % l];
    const auto p2 = points[(static_cast<int>(intPoint) + 1) % l];

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
