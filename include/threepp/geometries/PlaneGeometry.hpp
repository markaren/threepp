// https://github.com/mrdoob/three.js/blob/r129/src/geometries/PlaneGeometry.js

#ifndef THREEPP_PLANEGEOMETRY_HPP
#define THREEPP_PLANEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class PlaneGeometry: public BufferGeometry {

    public:
        const float width;
        const float height;

        PlaneGeometry(const PlaneGeometry&) = delete;

        [[nodiscard]] std::string type() const override {

            return "PlaneGeometry";
        }

        static std::shared_ptr<PlaneGeometry> create(
                float width = 1,
                float height = 1,
                unsigned int widthSegments = 1,
                unsigned int heightSegments = 1) {

            return std::shared_ptr<PlaneGeometry>(new PlaneGeometry(width, height, widthSegments, heightSegments));
        }

    protected:
        PlaneGeometry(float width, float height, unsigned int widthSegments, unsigned int heightSegments);
    };

}// namespace threepp

#endif//THREEPP_PLANEGEOMETRY_HPP
