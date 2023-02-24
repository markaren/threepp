// https://github.com/mrdoob/three.js/blob/r129/docs/api/en/geometries/WireframeGeometry.html

#ifndef THREEPP_WIREFRAMEGEOMETRY_HPP
#define THREEPP_WIREFRAMEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class WireframeGeometry: public BufferGeometry {

    public:
        static std::shared_ptr<WireframeGeometry> create(BufferGeometry& geometry) {

            return std::shared_ptr<WireframeGeometry>(new WireframeGeometry(geometry));
        }

    protected:
        explicit WireframeGeometry(BufferGeometry& geometry);
    };

}// namespace threepp

#endif//THREEPP_WIREFRAMEGEOMETRY_HPP
