// https://github.com/mrdoob/three.js/blob/r129/src/geometries/OctahedronGeometry.js

#ifndef THREEPP_OCTAHEDRONGEOMETRY_HPP
#define THREEPP_OCTAHEDRONGEOMETRY_HPP

#include "threepp/geometries/PolyhedronGeometry.hpp"

namespace threepp {

    class OctahedronGeometry: public PolyhedronGeometry {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<OctahedronGeometry> create(float radius = 1, unsigned int detail = 0);

    protected:
        explicit OctahedronGeometry(float radius, unsigned int detail);
    };

}// namespace threepp

#endif//THREEPP_OCTAHEDRONGEOMETRY_HPP
