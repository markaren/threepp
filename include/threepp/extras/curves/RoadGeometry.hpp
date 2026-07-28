// A road surface: an ALIGNMENT swept with a level cross-section.
//
// The road is not the authored curve offset sideways. It is the surface swept
// along a G1 alignment built from that curve — a biarc chain in plan and a
// second one over elevation (see RoadAlignment.hpp) — sampled at STATIONS
// chosen so neither the heading nor the grade breaks by more than a degree
// between neighbours.
//
// Every arc in the alignment is wider than the half-width by construction, so
// the inner edge of every bend has somewhere to be. That single property is
// what removes, rather than treats, the whole family of artefacts a swept road
// is prone to: there is no offset self-intersection to trim, no miter to widen,
// no apex to pin a cross-section to, and no fan to stitch across. None of those
// mechanisms is present in this file, and none should be added — a shape that
// needs one is a shape the alignment should not have handed over.
//
// The cost is that the road may sit a little off the curve at a bend the
// authored spline drew tighter than the road is wide. That is the trade the
// user asked for: a road that bends a little rather than one that folds. How
// far it bent, and how many bends it happened at, is reported (alignment().
// report()) rather than silently absorbed.
//
// Vertices are laid out one CROSS-SECTION at a time — left edge then right edge
// of station 0, then station 1, and so on — so a road is always an even number
// of vertices and always exactly `width` across at every one of them. UVs run
// u = arc length / uvLength along the road and v = 0 left, 1 right.
//
// A road whose two far-apart lobes overlap is out of scope: the pieces are each
// correct, their overlap is not resolved. That was true of every version of
// this file and is a property of sweeping, not of the alignment.

#ifndef THREEPP_ROADGEOMETRY_HPP
#define THREEPP_ROADGEOMETRY_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/core/Curve.hpp"
#include "threepp/extras/curves/RoadAlignment.hpp"

#include <array>
#include <memory>
#include <vector>

namespace threepp {

    class RoadGeometry: public BufferGeometry {

    public:
        struct Params {
            float width;
            // Seeds the alignment STARTS from. Refinement adds more until the
            // chain is within `tolerance` of the curve, and the stations the
            // surface is actually built at are chosen by angle — so this sets
            // where the fit begins, not how finely the road is tessellated.
            unsigned int divisions;
            // Metres of road per U tile.
            float uvLength;
            bool closed;

            // How far the alignment may sit from the authored curve, metres.
            float tolerance = 0.02f;
            // Angle one cross-section may break by against the next, radians.
            float stationAngle = 0.0174533f;
            // No bend tighter than the half-width times this.
            float minRadiusFactor = 1.05f;
            // Floor on the vertical curves, metres.
            float profileMinRadius = 10.f;

            explicit Params(float width = 1,
                            unsigned int divisions = 64,
                            float uvLength = 1,
                            bool closed = false);
        };

        const float width;

        [[nodiscard]] std::string type() const override;

        // What the surface was swept along, and what that cost the drawing.
        [[nodiscard]] const RoadAlignment& alignment() const { return alignment_; }

        // `path` is only read.
        static std::shared_ptr<RoadGeometry> create(const Curve3& path, const Params& params);

        static std::shared_ptr<RoadGeometry> create(const Curve3& path,
                                                    float width = 1,
                                                    unsigned int divisions = 64,
                                                    float uvLength = 1,
                                                    bool closed = false);

        // The road's collider: one CONVEX hull per station interval — the four
        // corners of that strip plus the same four pushed `thickness` down the
        // cross-sections' own normals. Boxes on a straight, wedges through a
        // bend, and consecutive hulls share a whole joint cross-section, top
        // and bottom, so there is no seam to fall through.
        //
        // Convex rather than a triangle mesh because a triangle mesh cannot be
        // attached to a moving actor: a road and a conveyor belt are the same
        // surface, and one of them has to be drivable.
        //
        // Read off the SURFACE, not off the alignment, so the collider is the
        // vertices the user is looking at — including on a road loaded from a
        // document, which carries its mesh and not its curve. Empty unless the
        // geometry is laid out in cross-sections, which is what this class
        // produces.
        [[nodiscard]] static std::vector<std::array<Vector3, 8>> hulls(const BufferGeometry& surface,
                                                                       float thickness);

    private:
        RoadGeometry(const Curve3& path, const Params& params);

        RoadAlignment alignment_;
    };

}// namespace threepp

#endif//THREEPP_ROADGEOMETRY_HPP
