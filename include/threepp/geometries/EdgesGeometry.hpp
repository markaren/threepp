// https://github.com/mrdoob/three.js/blob/r129/src/geometries/EdgesGeometry.js

#ifndef THREEPP_EDGESGEOMETRY_HPP
#define THREEPP_EDGESGEOMETRY_HPP

#include <threepp/core/BufferGeometry.hpp>

namespace threepp {

    class EdgesGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<EdgesGeometry> create(const BufferGeometry& geometry, float thresholdAngle = 90);

        [[nodiscard]] std::string type() const override;

    protected:
        EdgesGeometry(const BufferGeometry* geometry, float thresholdAngle);
    };

}// namespace threepp

#endif//THREEPP_EDGESGEOMETRY_HPP
