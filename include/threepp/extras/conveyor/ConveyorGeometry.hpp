// PhysX-free conveyor description + geometry: the waypoint path model (straight
// runs, exact circular-arc bends, per-segment surfaces), the belt/wall ribbon
// builders, roller / cleat / frame layout helpers and the procedural belt
// texture. Everything here is shared between the editor (authoring preview +
// generated meshes) and the physics runtime (ConveyorPhysics.hpp), so the
// colliders always match what is drawn. Deliberately free of PhysX so the
// library proper can build it; belt physics is created from the same spec on
// the sim side.
//
// All assets are first-party and procedural — the conveyor look (belt, rollers,
// cleats, side rails, legs, end drums) is generated, not loaded from models.

#ifndef THREEPP_CONVEYOR_GEOMETRY_HPP
#define THREEPP_CONVEYOR_GEOMETRY_HPP

#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class BufferGeometry;
    class DataTexture;

}// namespace threepp

namespace threepp::conveyor {

    // Per-segment surface kind. A segment is the span LEAVING a waypoint (to the
    // next one); by default it's a flat belt, with rollers / cleats opt-in PER
    // SEGMENT (not per path).
    enum class SegKind {
        Flat = 0,
        Rollers = 1,
        Cleats = 2
    };

    // A path waypoint. Normally a point on the centerline; if arcCenter is set,
    // this point is the CENTRE of a circular arc between its two neighbours
    // (start, end) — used for exact 90/180 degree horizontal bends instead of a
    // spline approximation. segKind is the surface of the segment leaving this
    // waypoint (unused on the last waypoint).
    struct Waypoint {
        Vector3 pos;
        bool arcCenter = false;
        SegKind segKind = SegKind::Flat;
    };

    // One conveyor: a centerline (waypoints) + its belt settings. This is the
    // resolved, self-contained description the physics runtime consumes — the
    // editor stores the same fields in userData (see editor::ConveyorConfig) and
    // resolves to this.
    struct ConveyorSpec {
        std::vector<Waypoint> waypoints;
        float width = 0.6f;
        float speed = 0.6f;   // m/s along travel; the runtime can scale it live
        bool reverse = false; // flip travel direction (and inlet end)
        bool smooth = true;   // spline through the (non-arc) waypoints
        // Separator: a collision-only vertical wall along the centerline (a
        // guide rail / lane divider) instead of a moving belt surface.
        bool separator = false;
        float wallHeight = 0.5f;
        // Shared tuning for whichever segments opt into rollers / cleats.
        float rollerRadius = 0.05f;
        float cleatHeight = 0.15f;
        float cleatSpacing = 0.6f;
        // Resample density (points per waypoint segment) for the smooth spline
        // and for arc tessellation.
        int samples = 12;
        // Generate the support frame (side rails, legs, end drums) — visual
        // only, but what makes a belt read as a machine instead of a ribbon.
        bool frame = true;
    };

    // Orientation for a belt segment a→b: local +X = travel direction (full 3D,
    // so sloped segments tilt), local +Z = horizontal width axis, local +Y =
    // belt normal. Shared by the preview geometry and the sim colliders so they
    // always match.
    Quaternion segmentOrientation(const Vector3& a, const Vector3& b);

    // One continuous, gap-free belt-surface ribbon of the given width following
    // a centerline polyline: one left/right vertex pair per point, each
    // cross-section edge shared between neighbouring quads so curves stay
    // watertight (independent per-segment quads would fan apart on the outside
    // of a bend). The width axis is horizontal and bisects turns (miter join);
    // sloped runs tilt with the tangent. UV.v carries cumulative arc length so
    // a texture scrolls uniformly along travel.
    std::shared_ptr<BufferGeometry> ribbonGeometry(const std::vector<Vector3>& centerline,
                                                   float width);

    // A vertical wall ribbon of the given height standing ON the centerline
    // (base at the centerline, top extruded straight up) — used for separators /
    // guide rails. Face normal is the horizontal lateral, so pair it with a
    // double-sided material. UV: u = arc length, v = height.
    std::shared_ptr<BufferGeometry> wallGeometry(const std::vector<Vector3>& centerline,
                                                 float height);

    // --- Rollers ---------------------------------------------------------------

    // A single conveyor roller: a cylinder lying ACROSS the belt (long axis =
    // width direction) whose top touches the conveying surface. `orientation`
    // maps a unit cylinder's local +Y (CylinderGeometry's axis) onto the width
    // axis, so a roller spins about its own +Y.
    struct Roller {
        Vector3 center;
        Quaternion orientation;
    };

    // Centre-to-centre spacing of rollers of a given radius: just over the
    // diameter, so the cylinders nearly touch — a tight roller bed.
    float rollerSpacing(float radius);

    // Lay rollers along a centerline: one cylinder every `spacing` metres of
    // arc length, axis = horizontal lateral, centre dropped `radius` below the
    // centerline so the roller TOP sits on the conveying surface. Half-spacing
    // margins keep rollers within the path.
    std::vector<Roller> rollerTransforms(const std::vector<Vector3>& centerline,
                                         float radius, float spacing);

    // --- Cleats ----------------------------------------------------------------

    // A conveyor cleat (a.k.a. flight): a thin bar standing ACROSS the belt that
    // catches product on an incline so it can't slide back.
    inline constexpr float kCleatThickness = 0.04f;// bar thickness along travel (metres)

    struct Cleat {
        Vector3 center;        // box centre: on the surface, raised height*0.5 along the normal
        Quaternion orientation;// segmentOrientation: local +X travel, +Y normal, +Z width
    };

    // Pose of a cleat at arc-length `s` along a centerline, folded by
    // `foldAngle` (radians) about its BASE edge (the width axis): 0 = standing
    // perpendicular, +-PI/2 = lying flat along travel. The base edge stays
    // anchored on the belt, so the bar pivots down flat at the ends rather than
    // sinking.
    void cleatPoseAt(const std::vector<Vector3>& centerline, float s, float height,
                     float foldAngle, Vector3& center, Quaternion& orientation);

    // Fold angle of a cleat at arc-length `s`: 0 across the carrying run,
    // easing to +-PI/2 within `rampLen` of each end so the bar lies flat at the
    // pulleys (a cleat models a closed band wrapping around them).
    float cleatFold(float s, float length, float rampLen);

    // Arc-length positions of cleats that EVENLY tile a run of length `length`
    // at ~`spacing` apart. The pitch derives from a whole bar count so every
    // gap — including across the wrap seam — stays equal.
    std::vector<float> cleatOffsets(float length, float spacing);

    // Lay cleats along a centerline, evenly tiled, each standing `height` above
    // the surface (fold angle 0 — the static preview arrangement).
    std::vector<Cleat> cleatTransforms(const std::vector<Vector3>& centerline,
                                       float height, float spacing);

    // --- Arcs + path resampling ------------------------------------------------

    // Circular-arc parameters for a bend defined by an arc-CENTRE waypoint: the
    // arc runs from A (the last centreline point) to B (the next waypoint)
    // around centre C in the horizontal (XZ) plane. `incoming` is the travel
    // direction arriving at A, used only to break the ~180 degree tie.
    struct Arc {
        bool valid = false;
        float a0 = 0.f;   // start angle atan2(z,x) of (A-C)
        float sweep = 0.f;// signed swept angle from A to B (radians)
        float radA = 0.f, radB = 0.f;
    };

    Arc computeArc(const Vector3& A, const Vector3& C, const Vector3& B,
                   const Vector3& incoming);

    // Resample the waypoints into a dense point list shared by the preview and
    // the sim. Any arc-centre present: straight runs between regular points and
    // a true circular arc at each arc-centre node. Else smooth: centripetal
    // Catmull-Rom through the points. Else: the raw polyline.
    std::vector<Vector3> resamplePath(const std::vector<Waypoint>& wps, bool smooth,
                                      int samplesPerSegment = 12);

    // A maximal run of consecutive same-kind segments, with its own dense
    // centerline. Runs share a boundary point with their neighbours, so a
    // flat→rollers change meets gap-free.
    struct PathRun {
        SegKind kind = SegKind::Flat;
        std::vector<Vector3> pts;
    };

    // Like resamplePath, but split into runs by per-segment kind (Waypoint
    // segKind of the segment's starting waypoint). Adjacent same-kind segments
    // merge; arc spans are always flat.
    std::vector<PathRun> resamplePathByKind(const std::vector<Waypoint>& wps, bool smooth,
                                            int samplesPerSegment = 12);

    // Point a given arc-length distance into a polyline; clamps to the last
    // point. Useful for spawn / inlet positions a bit onto the belt.
    Vector3 pointAlong(const std::vector<Vector3>& path, float dist);

    // --- Frame (first-party procedural "asset") --------------------------------

    // Proportions of the generated support frame, all derived from the belt
    // width so one knob scales the whole machine sensibly.
    struct FrameProfile {
        float railHeight = 0.10f;   // side-rail section height
        float railThickness = 0.05f;// side-rail section width
        float railTopOffset = 0.02f;// rail top above the conveying surface
        float legThickness = 0.06f; // square leg section
        float legSpacing = 2.0f;    // metres of arc length between leg pairs
        float drumRadius = 0.08f;   // end-drum (pulley) radius
        float minLegLength = 0.05f; // shorter than this and the leg is skipped

        // A profile scaled to a given belt width (wider belt, sturdier frame).
        static FrameProfile forWidth(float width);
    };

    // One side rail: a closed rectangular-section extrusion following the
    // centerline at horizontal offset `side`*(width/2 + railThickness/2)
    // (side = -1 left, +1 right), spanning from railTopOffset above the surface
    // down railHeight. Watertight along the path like the belt ribbon.
    std::shared_ptr<BufferGeometry> railGeometry(const std::vector<Vector3>& centerline,
                                                 float width, int side,
                                                 const FrameProfile& profile);

    // A vertical support leg: from under the rail straight down to `floorY`.
    struct Leg {
        Vector3 center;        // box centre (leg spans length about it, vertically)
        Quaternion orientation;// yaw so the section aligns with the path
        float length = 0.f;    // vertical extent
    };

    // Leg pairs every legSpacing metres of arc length (plus one pair near each
    // end), at both sides of the belt, dropping to `floorY`. Legs whose span
    // would be shorter than minLegLength are skipped.
    std::vector<Leg> legTransforms(const std::vector<Vector3>& centerline,
                                   float width, float floorY,
                                   const FrameProfile& profile);

    // End drums (pulleys): a cylinder across the belt at each end of an open
    // path, centre dropped so the belt surface is tangent to the drum top.
    // Same orientation convention as Roller (cylinder +Y = width axis).
    std::vector<Roller> endDrumTransforms(const std::vector<Vector3>& centerline,
                                          const FrameProfile& profile);

    // --- Belt texture ----------------------------------------------------------

    // Procedural modular-belt texture: a transverse groove per tile (reads
    // clearly as motion when scrolled along travel) plus a thin longitudinal
    // module line. Tile it via Repeat wrap; clone per belt so each scrolls
    // independently.
    std::shared_ptr<DataTexture> beltTexture();

    // Metres of belt per texture tile — the repeat the texture is designed for.
    inline constexpr float kBeltTileLength = 0.25f;

}// namespace threepp::conveyor

#endif// THREEPP_CONVEYOR_GEOMETRY_HPP
