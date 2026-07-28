// A sampled centerline consolidated into STRAIGHT and CIRCULAR-ARC pieces.
//
// For layouts that want ANALYTIC pieces rather than a polyline: a conveyor
// whose bends are driven as one rotating body, a path exported to a controller
// that speaks arcs, a collider tiled by wedges. The segmentation walks the
// samples and takes the LONGEST run a straight or a circle fits within
// `tolerance`, so a curve of hundreds of spans comes out as a handful of pieces
// whose swept region is known in closed form — a rectangle for a straight, an
// annulus for an arc (outer radius R + w/2, inner max(R - w/2, 0)).
//
// This is NOT what the editor's roads are built from. A fit is an
// approximation, and RoadGeometry follows the authored curve exactly — sample
// for sample, with its offsets trimmed. Consolidation changes the shape the
// user drew, however slightly, and leaves a curvature break at every joint.
//
// The fit is in the XZ plane with Y carried along as a per-piece LINEAR
// profile; a piece ends where the grade stops being linear, so a hill keeps its
// shape. A curve that loops vertically has no XZ projection to fit.

#ifndef THREEPP_ROADPATH_HPP
#define THREEPP_ROADPATH_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/math/Vector3.hpp"

#include <vector>

namespace threepp {

    // One piece of a consolidated centerline. A Straight runs start → end; an
    // Arc runs around `center` at `radius` from `startAngle` through `sweep`
    // (signed, radians, atan2(z, x) in the XZ plane), with its endpoints on the
    // sample chain so consecutive pieces meet exactly.
    struct RoadPrimitive {

        enum class Kind {
            Straight,
            Arc
        };

        Kind kind = Kind::Straight;
        Vector3 start;
        Vector3 end;
        // Arc only. The centre is a point in the XZ plane; its Y is unused and
        // kept at zero, since the height profile is carried by the endpoints.
        Vector3 center;
        float radius = 0.f;
        float startAngle = 0.f;
        float sweep = 0.f;

        // Length of the piece's centerline, grade included.
        [[nodiscard]] float length() const;
        // Length of its XZ projection — the radius the annulus is measured
        // against, and what the UV advances along.
        [[nodiscard]] float horizontalLength() const;

        // t runs 0..1 over the piece. The ends answer `start` and `end`
        // EXACTLY, which is what makes the chain watertight.
        [[nodiscard]] Vector3 pointAt(float t) const;
        // Unit tangent of the XZ projection, or zero where there is none (a
        // piece that only climbs).
        [[nodiscard]] Vector3 tangentAt(float t) const;
    };

    class RoadPath {

    public:
        struct Params {
            // How far a sample may sit from the piece fitted through it, in
            // metres — laterally in XZ and vertically against the linear grade.
            // Loose enough that a spline's gentle stretches consolidate, tight
            // enough that a real bend is its own arc.
            float tolerance;
            // How far a piece's tangent may sit from the sampled curve's at the
            // two samples it ENDS on, in radians. Position tolerance alone lets
            // a piece run to wherever its fit finally breaks, pointing somewhere
            // else by the time it gets there — and the two pieces meeting at
            // that sample then cross at a visible crease. Bounding the tangent
            // at the ends bounds the crease at twice this.
            float angleTolerance;

            explicit Params(float tolerance = 0.05f, float angleTolerance = 0.0873f);// 5 degrees
        };

        RoadPath() = default;

        // `samples` is an ordered centerline polyline; `closed` says its last
        // sample is its first. The pieces are chained: each starts at the
        // sample the one before it ended on.
        static RoadPath fromPoints(const std::vector<Vector3>& samples, bool closed,
                                   const Params& params = Params());

        // The same, sampling `curve` at `divisions` uniform steps — the
        // tessellation the editor's overlay and generated mesh already share.
        static RoadPath fromCurve(const Curve3& curve, unsigned int divisions, bool closed,
                                  const Params& params = Params());

        [[nodiscard]] const std::vector<RoadPrimitive>& primitives() const { return primitives_; }
        [[nodiscard]] bool closed() const { return closed_; }
        [[nodiscard]] bool empty() const { return primitives_.empty(); }
        // Chained length of every piece.
        [[nodiscard]] float length() const;

    private:
        std::vector<RoadPrimitive> primitives_;
        bool closed_ = false;
    };

}// namespace threepp

#endif//THREEPP_ROADPATH_HPP
