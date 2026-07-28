// A road surface built from a consolidated centerline (RoadPath).
//
// Where RibbonGeometry offsets a cross-section at every sample of a dense
// polyline — and has to clamp, pin or miter whatever that produces — this
// emits each piece's swept region directly:
//
//   straight : a rectangle, full width, always;
//   arc      : an annular sector strip, outer radius R + w/2 and inner radius
//              max(R - w/2, 0), stepped fine enough to read as smooth.
//
// A bend tighter than the half-width takes the inner radius to zero and the
// annulus becomes a PIE SECTOR whose inner vertices sit on the arc's centre.
// That is the correct swept region there — the piece keeps its full outer
// reach, the neighbouring pieces cover the inside of the corner, and no edge
// ever runs backward, so there is nothing to fold.
//
// Two vertices per cross-section, one quad per step, wound so the face normal
// points along the surface normal (+Y on the flat, tilted with the grade).
// Every cross-section is built in ITS OWN piece's exact frame — perpendicular
// along a straight, radial around an arc — and nothing is averaged across a
// joint: near the centre of a tight bend a degree of skew is most of a radius,
// and a rotated cross-section folds. The two cross-sections a joint carries sit
// on the same centreline point, and the quad between them covers the wedge the
// residual kink opens on the outside of the turn. UVs run u = arc length /
// uvLength along the road and v = 0..1 across it.
//
// Roads only: the fit behind RoadPath is an XZ one. A ribbon that loops
// vertically wants RibbonGeometry, which sweeps along anything.

#ifndef THREEPP_ROADGEOMETRY_HPP
#define THREEPP_ROADGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/curves/RoadPath.hpp"

namespace threepp {

    class RoadGeometry: public BufferGeometry {

    public:
        struct Params {
            float width;
            // Metres of road per U tile.
            float uvLength;
            // Largest angle one arc strip spans, in radians. The collider is
            // coarser on purpose; this is what the eye reads as a curve.
            float angularStep;

            explicit Params(float width = 1,
                            float uvLength = 1,
                            float angularStep = 0.1309f);// 7.5 degrees
        };

        const float width;

        [[nodiscard]] std::string type() const override;

        // `path` is only read.
        static std::shared_ptr<RoadGeometry> create(const RoadPath& path, const Params& params);

        static std::shared_ptr<RoadGeometry> create(const RoadPath& path,
                                                    float width = 1,
                                                    float uvLength = 1);

    private:
        RoadGeometry(const RoadPath& path, const Params& params);
    };

}// namespace threepp

#endif//THREEPP_ROADGEOMETRY_HPP
