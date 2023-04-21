
#include "threepp/geometries/CapsuleGeometry.hpp"

#include "threepp/extras/core/Path.hpp"

using namespace threepp;


namespace {

    std::vector<Vector2> generatePoints(float radius, float length, unsigned int capSegments) {

        Path path;
        path.absarc(0, -length / 2, radius, math::PI * 1.5f, 0);
        path.absarc(0, length / 2, radius, 0, math::PI * 0.5f);

        return path.getPoints(capSegments);
    }

}// namespace

CapsuleGeometry::CapsuleGeometry(const Params& params)
    : LatheGeometry(generatePoints(params.radius, params.length, params.capSegments), params.radialSegments),
      radius(params.radius),
      length(params.length) {}

std::string CapsuleGeometry::type() const {

    return "CapsuleGeometry";
}

std::shared_ptr<CapsuleGeometry> CapsuleGeometry::create(const CapsuleGeometry::Params& params) {

    return std::shared_ptr<CapsuleGeometry>(new CapsuleGeometry(params));
}

std::shared_ptr<CapsuleGeometry> CapsuleGeometry::create(float radius, float length, unsigned int capSegments, unsigned int radialSegments) {

    return create(Params(radius, length, capSegments, radialSegments));
}

CapsuleGeometry::Params::Params(float radius, float length, unsigned int capSegments, unsigned int radialSegments)
    : radius(radius), length(length),
      capSegments(capSegments), radialSegments(radialSegments) {}
