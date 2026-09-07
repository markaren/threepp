// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TorusGeometry.js

#ifndef THREEPP_TORUSGEOMETRY_HPP
#define THREEPP_TORUSGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class TorusGeometry: public BufferGeometry {

    public:
        struct Params {
            float radius;
            float tube;
            unsigned int radialSegments;
            unsigned int tubularSegments;
            float arc;

            explicit Params(float radius = 1,
                            float tube = 0.4f,
                            unsigned int radialSegments = 20,
                            unsigned int tubularSegments = 64,
                            float arc = math::TWO_PI)
                : radius(radius), tube(tube),
                  radialSegments(radialSegments), tubularSegments(tubularSegments), arc(arc) {}
        };

        // Construction parameters, kept so the geometry can be re-serialized in
        // three.js' compact parametric form (see ObjectExporter).
        const Params parameters;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<TorusGeometry> create(const Params& params);

        static std::shared_ptr<TorusGeometry> create(
                float radius = 1,
                float tube = 0.4f,
                unsigned int radialSegments = 20,
                unsigned int tubularSegments = 64,
                float arc = math::TWO_PI);

    protected:
        explicit TorusGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_TORUSGEOMETRY_HPP
