// https://github.com/mrdoob/three.js/blob/r129/src/geometries/PlaneGeometry.js

#ifndef THREEPP_PLANEGEOMETRY_HPP
#define THREEPP_PLANEGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class PlaneGeometry: public BufferGeometry {

    public:
        struct Params {
            float width;
            float height;

            unsigned int widthSegments;
            unsigned int heightSegments;

            explicit Params(float width = 1,
                            float height = 1,
                            unsigned int widthSegments = 1,
                            unsigned int heightSegments = 1);
        };

        const float width;
        const float height;

        PlaneGeometry(const PlaneGeometry&) = delete;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<PlaneGeometry> create(const Params& params);

        static std::shared_ptr<PlaneGeometry> create(
                float width = 1,
                float height = 1,
                unsigned int widthSegments = 1,
                unsigned int heightSegments = 1);

    protected:
        explicit PlaneGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_PLANEGEOMETRY_HPP
