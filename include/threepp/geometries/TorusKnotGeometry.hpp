// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TorusKnotGeometry.js

#ifndef THREEPP_TORUSKNOTGEOMETRY_HPP
#define THREEPP_TORUSKNOTGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class TorusKnotGeometry: public BufferGeometry {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<TorusKnotGeometry> create(
                float radius = 1,
                float tube = 0.4f,
                unsigned int tubularSegments = 64,
                unsigned int radialSegments = 16,
                unsigned int p = 2,
                unsigned int q = 3);

    protected:
        TorusKnotGeometry(float radius, float tube, unsigned int tubularSegments, unsigned int radialSegments, unsigned int p, unsigned int q);
    };

}// namespace threepp

#endif//THREEPP_TORUSKNOTGEOMETRY_HPP
