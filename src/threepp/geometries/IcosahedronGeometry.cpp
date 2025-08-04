
#include "threepp/geometries/IcosahedronGeometry.hpp"

#include <cmath>

using namespace threepp;

namespace {

    std::vector<float> generateVertices() {

        const auto t = (1.f + std::sqrt(5.f)) / 2.f;

        return {
                -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0,
                0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t,
                t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1};
    }

    std::vector<unsigned int> generateIndices() {

        return {
                0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
    }

}// namespace


IcosahedronGeometry::IcosahedronGeometry(float radius, unsigned int detail)
    : PolyhedronGeometry(generateVertices(), generateIndices(), radius, detail) {}

std::string IcosahedronGeometry::type() const {

    return "IcosahedronGeometry";
}

std::shared_ptr<IcosahedronGeometry> IcosahedronGeometry::create(float radius, unsigned int detail) {

    return std::shared_ptr<IcosahedronGeometry>(new IcosahedronGeometry(radius, detail));
}
