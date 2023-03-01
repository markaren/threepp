
#include "threepp/geometries/CapsuleGeometry.hpp"

#include "threepp/extras/core/Path.hpp"

using namespace threepp;


namespace {

    std::vector<Vector2> generatePoints(float radius, float length, unsigned int capSegments) {

        Path path;
        path.absarc(0, -length / 2, radius, math::PI * 1.5f, 0);
        path.absarc(0, length / 2, radius, 0,math::PI * 0.5f);

        return path.getPoints(capSegments);
    }

}// namespace

CapsuleGeometry::CapsuleGeometry(float radius, float length, unsigned int capSegments, unsigned int radialSegments)
    : LatheGeometry(generatePoints(radius, length, capSegments), radialSegments), radius(radius), length(length) {
}
