// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TubeGeometry.js

#ifndef THREEPP_TUBEGEOMETRY_HPP
#define THREEPP_TUBEGEOMETRY_HPP

#include <utility>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"
#include "threepp/math/MathUtils.hpp"

namespace threepp {

    class TubeGeometry: public BufferGeometry {

    public:
        struct Params {
            unsigned int tubularSegments;
            float radius;
            unsigned int radialSegments;
            bool closed;

            explicit Params(unsigned int tubularSegments = 64,
                            float radius = 1,
                            unsigned int radialSegments = 32,
                            bool closed = false);
        };

        const float radius;
        const std::shared_ptr<Curve3> path;

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<TubeGeometry> create(
                const std::shared_ptr<Curve3>& path,
                const Params& params);

        static std::shared_ptr<TubeGeometry> create(
                const std::shared_ptr<Curve3>& path,
                unsigned int tubularSegments = 64,
                float radius = 1,
                unsigned int radialSegments = 16,
                bool closed = false);

    private:
        Curve3::FrenetFrames frames;

        TubeGeometry(std::shared_ptr<Curve3> path, const Params& params);
    };

}// namespace threepp

#endif//THREEPP_TUBEGEOMETRY_HPP
