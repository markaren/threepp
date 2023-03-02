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
        const std::shared_ptr<Curve3> path;
        const float radius;

        [[nodiscard]] std::string type() const override {

            return "TubeGeometry";
        }

        static std::shared_ptr<TubeGeometry> create(
                const std::shared_ptr<Curve3>& path,
                unsigned int tubularSegments = 64,
                float radius = 1,
                unsigned int radialSegments = 16,
                bool closed = false) {

            return std::shared_ptr<TubeGeometry>(new TubeGeometry(path, tubularSegments, radius, radialSegments, closed));
        }

    private:
        Curve3::FrenetFrames frames;

        TubeGeometry(std::shared_ptr<Curve3> path, unsigned int tubularSegments, float radius, unsigned int radialSegments, bool closed);
    };

}// namespace threepp

#endif//THREEPP_TUBEGEOMETRY_HPP
