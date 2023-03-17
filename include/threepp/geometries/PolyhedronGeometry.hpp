// https://github.com/mrdoob/three.js/blob/r150/src/geometries/PolyhedronGeometry.js

#ifndef THREEPP_POLYHEDRONGEOMETRY_HPP
#define THREEPP_POLYHEDRONGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    // note: see https://github.com/mrdoob/three.js/pull/3302
    class PolyhedronGeometry: public BufferGeometry {

    public:
        [[nodiscard]] std::string type() const override;

    protected:
        PolyhedronGeometry(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, float radius, unsigned int detail);
    };

}// namespace threepp

#endif//THREEPP_POLYHEDRONGEOMETRY_HPP
