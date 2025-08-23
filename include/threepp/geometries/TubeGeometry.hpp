// https://github.com/mrdoob/three.js/blob/r129/src/geometries/TubeGeometry.js

#ifndef THREEPP_TUBEGEOMETRY_HPP
#define THREEPP_TUBEGEOMETRY_HPP


#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"
#include "threepp/math/MathUtils.hpp"

#include <variant>

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

        [[nodiscard]] std::string type() const override;

        [[nodiscard]] const FrenetFrames& getFrenetFrames() const {
            return frames_;
        }

        const Curve3* getPath() const {
            if (std::holds_alternative<Curve3*>(path_)) {
                return std::get<Curve3*>(path_);
            }
            return std::get<std::shared_ptr<Curve3>>(path_).get();
        }

        static std::shared_ptr<TubeGeometry> create(
                std::variant<Curve3*, std::shared_ptr<Curve3>> path,
                const Params& params);

        static std::shared_ptr<TubeGeometry> create(
                std::variant<Curve3*, std::shared_ptr<Curve3>> path,
                unsigned int tubularSegments = 64,
                float radius = 1,
                unsigned int radialSegments = 16,
                bool closed = false);

    private:
        FrenetFrames frames_;
        std::variant<Curve3*, std::shared_ptr<Curve3>> path_;

        TubeGeometry(std::variant<Curve3*, std::shared_ptr<Curve3>> path, const Params& params);
    };

}// namespace threepp

#endif//THREEPP_TUBEGEOMETRY_HPP
