
#include "threepp/geometries/OctahedronGeometry.hpp"

using namespace threepp;

namespace {

    inline std::vector<float> vertices() {
        return {1, 0, 0, -1, 0, 0, 0, 1, 0,
                0, -1, 0, 0, 0, 1, 0, 0, -1};
    }

    inline std::vector<unsigned int> indices() {
        return {0, 2, 4, 0, 4, 3, 0, 3, 5,
                0, 5, 2, 1, 2, 5, 1, 5, 3,
                1, 3, 4, 1, 4, 2};
    }

}// namespace

OctahedronGeometry::OctahedronGeometry(float radius, unsigned int detail)
    : PolyhedronGeometry(vertices(), indices(), radius, detail) {}

std::string OctahedronGeometry::type() const {

    return "OctahedronGeometry";
}

std::shared_ptr<OctahedronGeometry> OctahedronGeometry::create(float radius, unsigned int detail) {

    return std::shared_ptr<OctahedronGeometry>(new OctahedronGeometry(radius, detail));
}
