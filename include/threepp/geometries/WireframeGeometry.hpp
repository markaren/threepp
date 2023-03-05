// https://github.com/mrdoob/three.js/blob/r129/docs/api/en/geometries/WireframeGeometry.html

#ifndef THREEPP_WIREFRAMEGEOMETRY_HPP
#define THREEPP_WIREFRAMEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class WireframeGeometry: public BufferGeometry {

    public:
        [[nodiscard]] std::string type() const override {

            return "WireframeGeometry";
        }

        static std::shared_ptr<WireframeGeometry> create(const BufferGeometry& geometry) {

            return std::shared_ptr<WireframeGeometry>(new WireframeGeometry(geometry));
        }

    protected:
        explicit WireframeGeometry(const BufferGeometry& geometry);
    };

}// namespace threepp

#endif//THREEPP_WIREFRAMEGEOMETRY_HPP
