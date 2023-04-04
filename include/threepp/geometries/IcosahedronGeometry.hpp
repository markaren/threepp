// https://github.com/mrdoob/three.js/blob/r150/src/geometries/IcosahedronGeometry.js

#ifndef THREEPP_ICOSAHEDRONGEOMETRY_HPP
#define THREEPP_ICOSAHEDRONGEOMETRY_HPP

#include "threepp/geometries/PolyhedronGeometry.hpp"

#include <memory>

namespace threepp {

    class IcosahedronGeometry: public PolyhedronGeometry {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<IcosahedronGeometry> create(float radius = 1, unsigned int detail = 0);

    private:
        IcosahedronGeometry(float radius, unsigned int detail);
    };

}// namespace threepp

#endif//THREEPP_ICOSAHEDRONGEOMETRY_HPP
