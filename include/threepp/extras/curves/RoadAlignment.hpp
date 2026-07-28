// A road ALIGNMENT: the line a road is swept along, and the cross-sections it
// is swept with.
//
// A swept surface is only ever as good as the line under it. Offsetting a
// sampled curve sideways is NOT that line: inside a bend tighter than the
// offset the two edges run past the centre of curvature and cross, and every
// answer to that — miter, trim, pin, fan — is a repair to a shape that should
// not have been produced. This header produces the shape instead, so none of
// those mechanisms has anything left to do.
//
// PLAN. Consecutive (point, tangent) seeds taken off the authored curve are
// joined by a BIARC: the classical pair of circular arcs that meets both seeds'
// tangents exactly and meets itself tangentially in between. Tangent continuity
// is therefore a property of the CONSTRUCTION rather than of a fitting
// tolerance — a joint has no angle to hold under a threshold. Seeds are refined
// until the chain sits within `tolerance` of the curve, and the figure achieved
// is reported rather than assumed.
//
// MINIMUM RADIUS. No arc may be tighter than the half-width: an inner edge that
// reaches the centre of curvature is the apex, the pinch and the fan all three.
// When a solved arc is under the floor the offending SEEDS are relaxed —
// smoothed toward their neighbours — and the chain is SOLVED AGAIN, so it is
// still G1 afterwards. A radius is never patched in place. The authored spline
// is never rejected: the road bends a little instead, and how much it bent is
// reported.
//
// PROFILE. Elevation against plan station is a chain of its own, built by the
// same solver in the (station, height) plane. Grade is continuous, and a grade
// change is a vertical curve rather than the crease a piecewise-linear profile
// leaves at every join.
//
// STATIONS. Both chains' piece boundaries, plus enough subdivision that heading
// and grade each turn by less than `stationAngle` between neighbours. A
// station's cross-section is perpendicular to the tangent, level side to side
// (a road banks nowhere) and FULL WIDTH — with every radius over the half-width
// there is no case left in which it would not be.

#ifndef THREEPP_ROADALIGNMENT_HPP
#define THREEPP_ROADALIGNMENT_HPP

#include "threepp/extras/core/Curve.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstddef>
#include <vector>

namespace threepp {

    // A circular arc in a plane, or a straight when its curvature is zero.
    // Curvature is signed: positive turns left, i.e. toward (-tangent.y,
    // tangent.x).
    struct Arc2 {

        Vector2 start;
        Vector2 tangent;// unit
        float curvature = 0.f;
        float length = 0.f;

        [[nodiscard]] Vector2 pointAt(float s) const;
        [[nodiscard]] Vector2 tangentAt(float s) const;
        [[nodiscard]] Vector2 end() const { return pointAt(length); }
        [[nodiscard]] Vector2 endTangent() const { return tangentAt(length); }
        // Infinity on a straight.
        [[nodiscard]] float radius() const;
        // Nearest distance from `query` to the arc itself, endpoints included.
        [[nodiscard]] float distanceTo(const Vector2& query) const;
    };

    // A chain of arcs that is tangent-continuous BY CONSTRUCTION.
    class BiarcChain {

    public:
        struct Seed {
            Vector2 point;
            Vector2 tangent;// unit
        };

        struct Limits {
            // No arc may be tighter than this. Zero disables the clamp, which
            // is what the refinement pass fits with — a chain is measured
            // against the authored curve before it is allowed to bend away
            // from it.
            float minRadius = 0.f;
            // How far a relaxed seed moves toward the midpoint of its
            // neighbours, per pass. Smoothing drives a chain toward a straight
            // line, whose radius is infinite, so the clamp always terminates;
            // this only sets how fast.
            float relaxation = 0.3f;
            int maxPasses = 400;
            // The profile chain's abscissa IS plan station: relaxing it would
            // move a seed to a station it is not at. Only the ordinate moves.
            bool freezeX = false;
        };

        struct Relaxation {
            int seeds = 0;   // seeds the clamp moved
            int bends = 0;   // arcs under the floor when the chain was first solved
            float moved = 0.f;// the farthest any seed moved, in the plane's units
        };

        BiarcChain() = default;

        // `seeds` is consumed: the minimum-radius clamp relaxes it in place.
        // A closed chain wraps from the last seed back to the first and has no
        // endpoints to pin.
        static BiarcChain fit(std::vector<Seed> seeds, bool closed,
                              const Limits& limits, Relaxation* relaxation = nullptr);

        [[nodiscard]] const std::vector<Arc2>& pieces() const { return pieces_; }
        [[nodiscard]] bool empty() const { return pieces_.empty(); }
        [[nodiscard]] float length() const { return length_; }

        // Station of every seed: `seedStations()[i]` is where seed i sits. A
        // closed chain reports one more than it was given, the last being the
        // first seed come round again.
        [[nodiscard]] const std::vector<float>& seedStations() const { return seedStations_; }
        // Index of the first piece of the biarc leaving seed i. One entry per
        // seed station, so interval i owns pieces [seedPieces()[i],
        // seedPieces()[i + 1]).
        [[nodiscard]] const std::vector<std::size_t>& seedPieces() const { return seedPieces_; }

        // Station at which each piece begins; `length()` closes the list.
        [[nodiscard]] const std::vector<float>& pieceStations() const { return starts_; }

        [[nodiscard]] std::size_t pieceAt(float station, float& local) const;
        [[nodiscard]] Vector2 pointAt(float station) const;
        [[nodiscard]] Vector2 tangentAt(float station) const;
        [[nodiscard]] float curvatureAt(float station) const;

        // Nearest distance from `query` to pieces [first, last).
        [[nodiscard]] float distanceTo(const Vector2& query,
                                       std::size_t first, std::size_t last) const;
        // Tightest radius in the chain; infinity when every piece is straight.
        [[nodiscard]] float minRadius() const;
        // Largest angle between one piece's end tangent and the next piece's
        // start tangent, in radians. Zero by construction — a measurement, not
        // a tolerance.
        [[nodiscard]] float maxAngleBreak(bool closed) const;

    private:
        std::vector<Arc2> pieces_;
        std::vector<float> starts_;
        std::vector<float> seedStations_;
        std::vector<std::size_t> seedPieces_;
        float length_ = 0.f;
    };

    class RoadAlignment {

    public:
        struct Params {
            float width = 4.f;
            bool closed = false;
            // Seeds the authored curve starts with. Refinement adds more; this
            // is where it starts, not where it ends.
            unsigned int seeds = 64;
            // How far the fitted chain may sit from the authored curve, in
            // metres, before another seed goes in.
            float tolerance = 0.02f;
            // Angle a cross-section may break by against its neighbour, in
            // radians, in plan AND in elevation. One degree.
            float stationAngle = 0.0174533f;
            // The plan floor is the half-width times this. Strictly over 1, so
            // the inner edge of every bend has somewhere to be.
            float minRadiusFactor = 1.05f;
            // Floor on the vertical curves, in metres. Deliberately SMALL. What
            // a crest owes a driver is a continuous grade, which the profile
            // chain gives it at any radius; a floor beyond that is not fixing a
            // crease, it is overruling the heights the user drew — at 10 m this
            // took half a metre off an authored hill, visibly. Four metres
            // bounds what a genuine vertical cusp can do (v^2/R = 2.25 m/s^2 at
            // 3 m/s) and clears a hand-drawn hill without touching it.
            float profileMinRadius = 4.f;
            unsigned int maxSeeds = 600;
            unsigned int maxStations = 4000;
        };

        // One cross-section of the road.
        struct Station {
            float distance = 0.f;// along the alignment, in plan
            Vector3 point;
            Vector3 tangent;// unit, 3D
            Vector3 side;   // unit, HORIZONTAL: a road never banks
            Vector3 normal; // unit, tangent x side
        };

        struct Report {
            // Farthest the REFINED chain sits from the authored curve in plan,
            // before the radius clamp: what the seeding achieved.
            float fit = 0.f;
            // Farthest the FINAL alignment sits from the authored curve in 3D:
            // the fit plus whatever bending the radius floors cost.
            float deviation = 0.f;
            float planMinRadius = 0.f;
            float profileMinRadius = 0.f;
            // Arcs under the floor when the chain was first solved, and seeds
            // the clamp had to move to lift them over it.
            int bendsRelaxed = 0;
            int seedsRelaxed = 0;
            std::size_t seeds = 0;
            std::size_t planPieces = 0;
            std::size_t profilePieces = 0;
        };

        // `path` is only read.
        static RoadAlignment build(const Curve3& path, const Params& params);

        [[nodiscard]] const std::vector<Station>& stations() const { return stations_; }
        [[nodiscard]] const Report& report() const { return report_; }
        [[nodiscard]] const BiarcChain& plan() const { return plan_; }
        [[nodiscard]] const BiarcChain& profile() const { return profile_; }
        [[nodiscard]] float length() const { return plan_.length(); }

    private:
        // Elevation and grade at plan station `s`. The profile is a chain in
        // the (station, height) plane, so this inverts its own arc length back
        // onto the abscissa.
        void sampleProfile(float station, float& height, float& grade) const;

        BiarcChain plan_;
        BiarcChain profile_;
        std::vector<Station> stations_;
        Report report_;
    };

}// namespace threepp

#endif//THREEPP_ROADALIGNMENT_HPP
