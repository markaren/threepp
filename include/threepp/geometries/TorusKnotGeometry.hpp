// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TorusKnotGeometry.js

#ifndef THREEPP_TORUSKNOTGEOMETRY_HPP
#define THREEPP_TORUSKNOTGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"

namespace threepp {

    class TorusKnotGeometry: public BufferGeometry {

    public:
        struct Params {
            float radius;
            float tube;
            unsigned int tubularSegments;
            unsigned int radialSegments;
            unsigned int p;
            unsigned int q;

            explicit Params(float radius = 1,
                            float tube = 0.4f,
                            unsigned int tubularSegments = 64,
                            unsigned int radialSegments = 16,
                            unsigned int p = 2,
                            unsigned int q = 3)
                : radius(radius), tube(tube),
                  tubularSegments(tubularSegments), radialSegments(radialSegments),
                  p(p), q(q) {}
        };

        // Construction parameters, kept so the geometry can be re-serialized in
        // three.js' compact parametric form (see ObjectExporter).
        const Params parameters;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<TorusKnotGeometry> create(const Params& params);

        static std::shared_ptr<TorusKnotGeometry> create(
                float radius = 1,
                float tube = 0.4f,
                unsigned int tubularSegments = 64,
                unsigned int radialSegments = 16,
                unsigned int p = 2,
                unsigned int q = 3);

    protected:
        explicit TorusKnotGeometry(const Params& params);
    };

}// namespace threepp

#endif//THREEPP_TORUSKNOTGEOMETRY_HPP
