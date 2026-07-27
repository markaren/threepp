// A flat ribbon swept along a curve: roads, paths, rails, belts.
//
// Where TubeGeometry sweeps a circle along the curve's Frenet frame, this
// sweeps a horizontal SEGMENT along a frame that is level side to side. Each
// SPAN has a direction and a sideways vector
//
//     side = normalize(cross(up, direction)),  up = (0, 1, 0)
//
// in the curve's OWN space — the ribbon is built where the curve is, so a
// rotated parent rotates the ribbon with it rather than twisting it back to
// world level. That is what makes a climbing curve produce a road that banks
// nowhere: it rises with the grade but never rolls.
//
// A cross-section belongs to the two spans meeting at it, not to one sample:
// it takes their BISECTOR and is widened by the miter factor 1/cos(theta/2)
// (clamped, so a hairpin does not spike), which puts its two vertices exactly
// where the neighbouring spans' offset edges cross. Per-sample side vectors
// instead let each span's edge overshoot into the next, and a corner tighter
// than the half-width folds the ribbon back over itself.
//
// Two vertices per sample, one quad per span, wound so the face normal points
// along +up. UVs run u = arcLength / uvLength along the ribbon and v = 0..1
// across it, so a tiling texture repeats every `uvLength` metres however the
// curve is tessellated.

#ifndef THREEPP_RIBBONGEOMETRY_HPP
#define THREEPP_RIBBONGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"

namespace threepp {

    class RibbonGeometry: public BufferGeometry {

    public:
        struct Params {
            float width;
            // Spans along the curve. divisions + 1 cross-sections are emitted.
            unsigned int divisions;
            // Metres of curve per U tile.
            float uvLength;
            bool closed;

            explicit Params(float width = 1,
                            unsigned int divisions = 64,
                            float uvLength = 1,
                            bool closed = false);
        };

        const float width;

        [[nodiscard]] std::string type() const override;

        // `path` is only read.
        static std::shared_ptr<RibbonGeometry> create(const Curve3& path, const Params& params);

        static std::shared_ptr<RibbonGeometry> create(const Curve3& path,
                                                      float width = 1,
                                                      unsigned int divisions = 64,
                                                      float uvLength = 1,
                                                      bool closed = false);

    private:
        RibbonGeometry(const Curve3& path, const Params& params);
    };

}// namespace threepp

#endif//THREEPP_RIBBONGEOMETRY_HPP
