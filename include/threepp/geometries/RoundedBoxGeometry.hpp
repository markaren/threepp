// https://github.com/mrdoob/three.js/blob/dev/examples/jsm/geometries/RoundedBoxGeometry.js

#ifndef THREEPP_ROUNDED_BOX_GEOMETRY_HPP
#define THREEPP_ROUNDED_BOX_GEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    // A box with rounded edges and corners. Every face keeps a flat middle,
    // and each edge is a quarter circle of the given radius made of `segments`
    // steps. The geometry is non-indexed, as in three.js.
    class RoundedBoxGeometry: public BufferGeometry {

    public:
        struct Params {
            float width;
            float height;
            float depth;

            unsigned int segments;
            float radius;

            explicit Params(float width = 1,
                            float height = 1,
                            float depth = 1,
                            unsigned int segments = 2,
                            float radius = 0.1f);
        };

        // Construction parameters, with the radius as used: never more than half
        // the shortest side.
        const Params parameters;

        [[nodiscard]] const std::string& type() const override;

        static std::shared_ptr<RoundedBoxGeometry> create(const Params& params);

        static std::shared_ptr<RoundedBoxGeometry> create(float width = 1, float height = 1, float depth = 1, unsigned int segments = 2, float radius = 0.1f);

    protected:
        explicit RoundedBoxGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_ROUNDED_BOX_GEOMETRY_HPP
