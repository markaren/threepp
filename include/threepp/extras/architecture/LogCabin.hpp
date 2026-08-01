// Procedural log-cabin generator (CPU).
//
// Builds a complete Scandinavian/North-American style log cabin — stacked
// round-log walls with notched, protruding corner ends, log gable walls, a
// shingled gable roof with a wall dormer, a full-length covered porch with
// posts/railing/steps, trimmed multi-pane windows, a panelled door, a rubble
// foundation and a flue — as a handful of merged BufferGeometries.
//
// Design goals
//   * SELF-CONTAINED. No external assets: every texture is generated in-engine
//     (see CabinTextures.hpp), so the cabin can be dropped into any demo.
//   * BACKEND-AGNOSTIC. Nothing here touches a renderer. The output is plain
//     geometry + MeshStandardMaterials, so it renders identically on the GL and
//     Vulkan paths; a host demo may swap the materials wholesale.
//   * FEW DRAW CALLS. Everything is merged into one geometry per material
//     bucket (~10 meshes for the whole building), so an instance costs nothing
//     meaningful in a large outdoor scene.
//   * PARAMETRIC. Footprint, log size, roof pitch, porch and every opening are
//     driven by CabinParams, and cabinMetrics() reports the derived heights a
//     host needs in order to sit the building on terrain.
//
// Coordinate convention: +Y is up. The building is centred on the origin in
// plan; the ridge runs along X, so the ±X walls are the GABLE ends and the ±Z
// walls carry the eaves. The porch is on the +Z ("front") side. y = 0 is grade
// — the foundation extends below it so the cabin can be sunk into a slope.

#ifndef THREEPP_EXTRAS_ARCHITECTURE_LOGCABIN_HPP
#define THREEPP_EXTRAS_ARCHITECTURE_LOGCABIN_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/architecture/CabinTextures.hpp"
#include "threepp/extras/core/Shape.hpp"
#include "threepp/geometries/ExtrudeGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace threepp::architecture {

    // ── Wall identification ──────────────────────────────────────────────
    //
    // Front/Back carry the eaves (long walls, parallel to the ridge); Left and
    // Right are the gable ends. An opening's `center` is a world coordinate on
    // the wall's own horizontal axis: X for Front/Back, Z for Left/Right.
    enum class WallSide {
        Front = 0,// +Z, the porch side
        Back = 1, // -Z
        Left = 2, // -X gable end
        Right = 3 // +X gable end
    };

    enum class OpeningKind {
        Window = 0,
        Door = 1
    };

    struct Opening {
        WallSide side = WallSide::Front;
        float center = 0.f;    // along the wall (world X or world Z)
        float width = 1.20f;   // clear width of the hole
        float sillHeight = 1.00f;// bottom of the hole above the floor deck
        float height = 1.55f;  // clear height of the hole
        OpeningKind kind = OpeningKind::Window;
        // Lights across, PER SASH. A tall opening becomes a double-hung pair
        // split at a meeting rail, so `panesX = 3` on a tall window reads as
        // nine-over-nine.
        int panesX = 3;
        // Lights high per sash. 0 = derive from panesX so the individual
        // lights come out SQUARE, which is what makes a divided-light sash
        // read as joinery instead of as an arbitrary lattice.
        int panesY = 0;
        bool shutters = false;
    };

    // ── Configurator-facing knobs ────────────────────────────────────────
    struct CabinParams {
        unsigned int seed = 7;

        // ── Footprint & walls ────────────────────────────────────────────
        float length = 15.0f;     // X — along the ridge (log centreline span)
        float depth = 8.0f;       // Z — gable-to-gable depth
        // Floor deck → top plate. Snapped up to a whole log course. Tall
        // enough that the porch roof, which hangs off the main eave, still
        // clears head height at the posts — see the porch section.
        float wallHeight = 3.50f;
        float floorHeight = 0.80f;// floor deck above grade

        // ── Logs ─────────────────────────────────────────────────────────
        float logRadius = 0.17f;// MINIMUM radius; jitter only ever adds to it
        // Vertical OVERLAP between consecutive courses, as a fraction of the
        // log radius. Courses alternate direction, so a given wall's own logs
        // land 2 * rise apart; that spacing must stay strictly BELOW a log
        // diameter or the wall is a set of floating cylinders with daylight
        // (and the dark backing shell) showing between every pair. Positive
        // overlap makes neighbouring cylinders interpenetrate, and the
        // intersection reads as the scribed groove of a real chinkless wall.
        float courseOverlap = 0.085f;
        float cornerOverhang = 0.40f;// how far notched log ends stick past a corner
        int logRadialSegments = 14;
        float logRadiusJitter = 0.13f;// per-log radius variation (added, never subtracted)
        float logSag = 0.012f;      // mid-span bow, world units
        int logAxialSegments = 5;   // stations along a log (>1 to show the bow)

        // ── Roof ─────────────────────────────────────────────────────────
        float roofPitchDeg = 38.f;
        float eaveOverhang = 0.42f; // past the ±Z walls
        float rakeOverhang = 0.45f; // past the ±X gable walls
        float roofThickness = 0.17f;
        float fasciaHeight = 0.14f;

        // ── Wall dormer (gable rising out of the front roof plane) ───────
        bool dormer = true;
        float dormerCenterX = -4.30f;
        float dormerWidth = 5.00f;
        float dormerRiseFrac = 0.82f;// fraction of the main eave→ridge rise
        // Sized to stay a SINGLE casement after snapping to the dormer's own
        // course grid. A dormer light is a small window; run it tall enough to
        // trip the double-hung split and it reads as a full first-floor sash
        // shrunk into the roof.
        float dormerWindowWidth = 0.90f;
        float dormerWindowHeight = 0.85f;
        float dormerWindowSill = 0.50f;// above the dormer's own base

        // ── Porch ────────────────────────────────────────────────────────
        bool porch = true;
        float porchDepth = 2.55f;   // out from the front wall centreline
        float porchStartX = -7.50f; // extent along X
        float porchEndX = 7.50f;
        float porchRoofPitchDeg = 14.f;
        float porchEaveOverhang = 0.30f;
        float porchRoofThickness = 0.13f;
        float postSize = 0.17f;
        float postSpacing = 2.15f;
        float beamHeight = 0.22f;
        float railHeight = 0.98f;
        float balusterSize = 0.055f;
        float balusterSpacing = 0.135f;
        float deckPlankWidth = 0.140f;
        float deckPlankGap = 0.018f;
        float stepsCenterX = 1.60f;
        float stepsWidth = 1.70f;
        int stepCount = 3;

        // Bake the low-frequency occlusion term into the vertex colours (see
        // the end of buildCabinGeometry). Leave on unless the host renderer
        // supplies its own AO and would double up.
        bool bakeOcclusion = true;

        // ── Foundation ───────────────────────────────────────────────────
        float foundationInset = 0.06f;// inside the log face
        float foundationDepth = 0.70f;// below grade, so it can be sunk into terrain

        // ── Flue ─────────────────────────────────────────────────────────
        bool flue = true;
        float flueX = 4.60f;
        float flueZ = -0.90f;
        float flueRadius = 0.115f;
        float flueRise = 1.40f;// above the roof surface where it exits

        // ── Porch sconces (small emissive lamps flanking the door) ───────
        bool sconces = true;

        // ── Texture tiling (world metres per texture tile) ───────────────
        float logTileLength = 3.10f;// along a log
        float shingleTileU = 1.25f; // across the roof
        float shingleTileV = 0.90f; // up the slope
        float timberTile = 1.30f;
        float stoneTile = 1.15f;
        unsigned int textureSize = 512;

        // ── Albedo hints (sRGB) ──────────────────────────────────────────
        std::array<float, 3> logColor = {0.455f, 0.268f, 0.128f};
        std::array<float, 3> logEndColor = {0.520f, 0.380f, 0.212f};
        std::array<float, 3> shingleColor = {0.300f, 0.213f, 0.152f};
        std::array<float, 3> timberColor = {0.480f, 0.325f, 0.185f};
        std::array<float, 3> stoneColor = {0.400f, 0.385f, 0.360f};
        std::array<float, 3> trimColor = {0.870f, 0.860f, 0.830f};
        // The backing behind the logs is only ever seen in the scribed groove
        // between courses, where it stands in for chinking in deep shadow. It
        // has to be very dark: under a bright sky even a dark brown lifts into
        // a visible grey band between every course and the wall stops reading
        // as stacked solid timber.
        std::array<float, 3> shellColor = {0.048f, 0.036f, 0.028f};

        // ── Openings ─────────────────────────────────────────────────────
        // Left empty → defaultOpenings() supplies a layout that matches the
        // proportions above. Any non-empty list replaces it entirely.
        std::vector<Opening> openings;
    };

    // Derived dimensions a host demo needs for placement (terrain flattening,
    // camera framing, prop placement, collision proxies).
    struct CabinMetrics {
        float gradeY = 0.f;      // y of the ground the cabin is authored against
        float floorY = 0.f;      // porch/interior deck level
        float eaveY = 0.f;       // top plate — where the roof meets the wall
        float ridgeY = 0.f;      // top of the roof surface at the ridge
        float halfLength = 0.f;  // log-centreline half extents
        float halfDepth = 0.f;
        float roofHalfLength = 0.f;// including the rake overhang
        float roofHalfDepth = 0.f; // including the eave overhang
        float porchOuterZ = 0.f;   // front face of the porch posts
        int wallCourses = 0;
        // Top of the flue, in cabin-local space — where a host anchors a smoke
        // emitter. Remember it is LOCAL: transform it by the cabin's world
        // matrix, or a rotated cabin will smoke from thin air beside itself.
        Vector3 flueTip;
    };

    // ── Internal mesh accumulation ───────────────────────────────────────
    namespace detail {

        constexpr float TWO_PI = 6.28318530717959f;
        constexpr float DEG = 0.01745329251994f;

        // Accumulates position/normal/uv/color into one indexed geometry.
        // Every vertex carries a colour so the generator can bake per-part
        // tonal variation (per-log stain drift, per-plank weathering) without
        // needing a separate material or texture per piece.
        class MeshBuilder {

        public:
            Vector3 tint{1.f, 1.f, 1.f};// applied to every vertex pushed

            [[nodiscard]] bool empty() const { return indices_.empty(); }

            // a,b,c,d must be CCW as seen from OUTSIDE the surface.
            void quad(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d,
                      const Vector2& ta, const Vector2& tb, const Vector2& tc, const Vector2& td) {
                Vector3 e1, e2, n;
                e1.subVectors(b, a);
                e2.subVectors(d, a);
                n.crossVectors(e1, e2);
                if (n.lengthSq() < 1e-16f) return;// degenerate
                n.normalize();
                const auto base = static_cast<unsigned int>(positions_.size() / 3);
                push(a, n, ta);
                push(b, n, tb);
                push(c, n, tc);
                push(d, n, td);
                indices_.insert(indices_.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
            }

            void tri(const Vector3& a, const Vector3& b, const Vector3& c,
                     const Vector2& ta, const Vector2& tb, const Vector2& tc) {
                Vector3 e1, e2, n;
                e1.subVectors(b, a);
                e2.subVectors(c, a);
                n.crossVectors(e1, e2);
                if (n.lengthSq() < 1e-16f) return;
                n.normalize();
                const auto base = static_cast<unsigned int>(positions_.size() / 3);
                push(a, n, ta);
                push(b, n, tb);
                push(c, n, tc);
                indices_.insert(indices_.end(), {base, base + 1, base + 2});
            }

            // Axis-aligned box. `tex` is texture tiles per world metre; the UV
            // origin is the box's own min corner so neighbouring boxes of the
            // same size do not all show the identical patch.
            void box(const Vector3& c, const Vector3& h, float tex,
                     float uPhase = 0.f, float vPhase = 0.f) {
                if (h.x <= 0.f || h.y <= 0.f || h.z <= 0.f) return;
                const float x0 = c.x - h.x, x1 = c.x + h.x;
                const float y0 = c.y - h.y, y1 = c.y + h.y;
                const float z0 = c.z - h.z, z1 = c.z + h.z;
                const float du = 2.f * h.x * tex, dv = 2.f * h.y * tex, dw = 2.f * h.z * tex;
                const Vector2 o{uPhase, vPhase};
                auto U = [&](float u, float v) { return Vector2{o.x + u, o.y + v}; };
                // +X / -X : u follows Z, v follows Y
                quad({x1, y0, z1}, {x1, y0, z0}, {x1, y1, z0}, {x1, y1, z1}, U(0, 0), U(dw, 0), U(dw, dv), U(0, dv));
                quad({x0, y0, z0}, {x0, y0, z1}, {x0, y1, z1}, {x0, y1, z0}, U(0, 0), U(dw, 0), U(dw, dv), U(0, dv));
                // +Y / -Y : u follows X, v follows Z
                quad({x0, y1, z1}, {x1, y1, z1}, {x1, y1, z0}, {x0, y1, z0}, U(0, 0), U(du, 0), U(du, dw), U(0, dw));
                quad({x0, y0, z0}, {x1, y0, z0}, {x1, y0, z1}, {x0, y0, z1}, U(0, 0), U(du, 0), U(du, dw), U(0, dw));
                // +Z / -Z : u follows X, v follows Y
                quad({x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}, U(0, 0), U(du, 0), U(du, dv), U(0, dv));
                quad({x1, y0, z0}, {x0, y0, z0}, {x0, y1, z0}, {x1, y1, z0}, U(0, 0), U(du, 0), U(du, dv), U(0, dv));
            }

            // Solid formed by extruding a planar quad straight DOWN in Y.
            // `top` must be CCW seen from above. Used for every roof plane, so
            // the eave/rake edges come out vertical — which is what a fascia
            // board actually is.
            void slab(const std::array<Vector3, 4>& top, const std::array<Vector2, 4>& topUV,
                      float thickness, float sideTex) {
                std::array<Vector3, 4> bot = top;
                for (auto& v : bot) v.y -= thickness;
                quad(top[0], top[1], top[2], top[3], topUV[0], topUV[1], topUV[2], topUV[3]);
                quad(bot[3], bot[2], bot[1], bot[0], topUV[3], topUV[2], topUV[1], topUV[0]);
                for (int i = 0; i < 4; ++i) {
                    const int j = (i + 1) % 4;
                    const float len = top[i].distanceTo(top[j]) * sideTex;
                    const float t = thickness * sideTex;
                    quad(bot[i], bot[j], top[j], top[i], {0.f, 0.f}, {len, 0.f}, {len, t}, {0.f, t});
                }
            }

            // Triangular prism: triangle abc swept by `offset`.
            void triPrism(const Vector3& a, const Vector3& b, const Vector3& c,
                          const Vector3& offset, float tex) {
                Vector3 e1, e2, n;
                e1.subVectors(b, a);
                e2.subVectors(c, a);
                n.crossVectors(e1, e2);
                if (n.lengthSq() < 1e-14f) return;
                // Order the base so its normal opposes the sweep direction; the
                // swept copy then faces along it and the side windings below are
                // consistent without a second case.
                Vector3 A = a, B = b, C = c;
                if (n.dot(offset) > 0.f) std::swap(B, C);
                Vector3 A2, B2, C2;
                A2.copy(A).add(offset);
                B2.copy(B).add(offset);
                C2.copy(C).add(offset);
                auto uv = [&](const Vector3& p) { return Vector2{p.x * tex + p.z * tex, p.y * tex}; };
                tri(A, B, C, uv(A), uv(B), uv(C));
                tri(A2, C2, B2, uv(A2), uv(C2), uv(B2));
                auto side = [&](const Vector3& p, const Vector3& q, const Vector3& q2, const Vector3& p2) {
                    quad(p, q, q2, p2, uv(p), uv(q), uv(q2), uv(p2));
                };
                side(B, A, A2, B2);
                side(C, B, B2, C2);
                side(A, C, C2, A2);
            }

            // Round log / pipe from `a` to `b`.
            //
            // `radiusB < 0` means "same as radiusA"; otherwise the log TAPERS,
            // as a real one does from butt to tip. `bow` displaces mid-span in
            // world space. Both matter more than they look: a wall of
            // identical dead-straight cylinders reads as extruded plastic
            // pipe, and the eye picks that up long before it picks up texture.
            // Caps go into `capOut` so the sawn end grain can carry its own
            // material; pass nullptr for a pipe whose ends are hidden.
            void tube(const Vector3& a, const Vector3& b, float radius, float radiusB,
                      int radialSegs, int axialSegs, const Vector3& bow,
                      float uPerMetre, MeshBuilder* capOut, float uPhase = 0.f,
                      float capUvSeed = 0.f) {
                Vector3 axis;
                axis.subVectors(b, a);
                const float len = axis.length();
                if (len < 1e-5f || radius <= 0.f) return;
                axis.divideScalar(len);

                Vector3 ref{0.f, 1.f, 0.f};
                if (std::abs(axis.dot(ref)) > 0.95f) ref.set(1.f, 0.f, 0.f);
                Vector3 P, Q;
                P.crossVectors(ref, axis).normalize();
                Q.crossVectors(axis, P).normalize();

                const int R = std::max(4, radialSegs);
                const int A = std::max(1, axialSegs);
                const auto base = static_cast<unsigned int>(positions_.size() / 3);

                const float rA = radius;
                const float rB = (radiusB < 0.f) ? radius : radiusB;

                auto centreAt = [&](float t) {
                    Vector3 p;
                    p.copy(a).addScaledVector(axis, t * len);
                    // Parabolic bow, zero at both ends.
                    p.addScaledVector(bow, 4.f * t * (1.f - t));
                    return p;
                };
                auto radiusAt = [&](float t) { return rA + (rB - rA) * t; };

                for (int i = 0; i <= A; ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(A);
                    const Vector3 ctr = centreAt(t);
                    const float rr = radiusAt(t);
                    const float u = uPhase + t * len * uPerMetre;
                    for (int j = 0; j <= R; ++j) {
                        const float ang = static_cast<float>(j) / static_cast<float>(R) * TWO_PI;
                        const float ca = std::cos(ang), sa = std::sin(ang);
                        Vector3 n{P.x * ca + Q.x * sa, P.y * ca + Q.y * sa, P.z * ca + Q.z * sa};
                        Vector3 p;
                        p.copy(ctr).addScaledVector(n, rr);
                        push(p, n, {u, static_cast<float>(j) / static_cast<float>(R)});
                    }
                }
                const auto stride = static_cast<unsigned int>(R + 1);
                for (int i = 0; i < A; ++i) {
                    const unsigned int r0 = base + static_cast<unsigned int>(i) * stride;
                    const unsigned int r1 = r0 + stride;
                    for (int j = 0; j < R; ++j) {
                        const unsigned int p0 = r0 + static_cast<unsigned int>(j);
                        const unsigned int p1 = p0 + 1;
                        const unsigned int p2 = r1 + static_cast<unsigned int>(j);
                        const unsigned int p3 = p2 + 1;
                        // CCW seen from OUTSIDE. The frame is right-handed with
                        // cross(P,Q) == axis, so the angular tangent crossed
                        // into the axis gives the outward radial normal — wind
                        // the other way and every log is back-face culled,
                        // leaving only the backing shell visible.
                        indices_.insert(indices_.end(), {p0, p1, p2, p1, p3, p2});
                    }
                }

                if (capOut) {
                    // Independent grain rotation per END — the two ends of one
                    // log are different cuts through different growth.
                    capOut->tint = tint;
                    capOut->disc(centreAt(0.f), P, Q, axis, rA, R, true, capUvSeed * TWO_PI);
                    capOut->disc(centreAt(1.f), P, Q, axis, rB, R, false,
                                 std::fmod(capUvSeed * 1.6180339f, 1.f) * TWO_PI);
                }
            }

            // Flat disc, used for sawn log ends. UVs map the disc into the unit
            // square so an end-grain texture lands centred on it.
            //
            // `uvRotation` spins the texture about the disc centre. Without it
            // every sawn end in the building shows the same pith in the same
            // place with its heart check pointing the same way — a column of
            // identical stamps at every corner notch, which is the loudest
            // repeat on the whole model. Rotating carries the off-centre pith
            // around with it, so each end reads as a different log.
            void disc(const Vector3& centre, const Vector3& P, const Vector3& Q,
                      const Vector3& axis, float radius, int segs, bool facingBack,
                      float uvRotation = 0.f) {
                const int R = std::max(4, segs);
                Vector3 n = axis;
                if (facingBack) n.multiplyScalar(-1.f);
                const auto base = static_cast<unsigned int>(positions_.size() / 3);
                push(centre, n, {0.5f, 0.5f});
                for (int j = 0; j <= R; ++j) {
                    const float ang = static_cast<float>(j) / static_cast<float>(R) * TWO_PI;
                    const float ca = std::cos(ang), sa = std::sin(ang);
                    const float ua = std::cos(ang + uvRotation), va = std::sin(ang + uvRotation);
                    Vector3 p;
                    p.copy(centre)
                            .addScaledVector(P, radius * ca)
                            .addScaledVector(Q, radius * sa);
                    push(p, n, {0.5f + 0.5f * ua, 0.5f + 0.5f * va});
                }
                for (int j = 0; j < R; ++j) {
                    const unsigned int i0 = base + 1 + static_cast<unsigned int>(j);
                    const unsigned int i1 = i0 + 1;
                    if (facingBack)
                        indices_.insert(indices_.end(), {base, i1, i0});
                    else
                        indices_.insert(indices_.end(), {base, i0, i1});
                }
            }

            // Merge an EXISTING BufferGeometry (from ExtrudeGeometry, Lathe,
            // Shape, a loaded mesh, ...) into this bucket, transformed by
            // `basis` + `origin`.
            //
            // This is the escape hatch from box-and-cylinder land: anything
            // whose silhouette is a curve — a sawn corbel, a moulded gutter
            // profile, a scrolled bracket — is drawn once as a 2-D Shape and
            // extruded, then folded in here so it still costs no extra draw
            // call. `basis` columns map the source X/Y/Z axes; pass a scaled
            // basis to size the part — including a non-uniform one, since
            // normals go through the proper inverse-transpose below. A basis
            // with negative determinant mirrors the part; winding is left
            // alone, so mirror in PAIRS (flip two axes) to stay right-handed.
            void append(const BufferGeometry& src, const Vector3& origin,
                        const Vector3& basisX, const Vector3& basisY, const Vector3& basisZ,
                        float texScale = 1.f) {
                const auto* pos = src.getAttribute<float>("position");
                if (!pos) return;
                // ExtrudeGeometry (like most three.js primitive builders that
                // emit per-face data) is NON-INDEXED. Requiring an index here
                // silently dropped every extruded part on the floor.
                const auto* idx = src.getIndex();
                const auto* nrm = src.getAttribute<float>("normal");
                const auto* uv = src.getAttribute<float>("uv");
                const auto& pa = pos->array();
                const auto base = static_cast<unsigned int>(positions_.size() / 3);
                const int n = pos->count();

                // Normals transform by the INVERSE-TRANSPOSE, which is only the
                // basis itself when the basis is orthogonal. Reusing a scaled
                // basis — which the contract above invites — tilts normals off
                // the surface: squash a part in Y and its normals lean the
                // wrong way, which reads as subtly wrong shading rather than as
                // anything obviously broken. The inverse is a single 3x3 PER
                // PART, not per vertex, so it costs nothing worth saving.
                // A degenerate basis inverts to the zero matrix; the per-vertex
                // fallback below then catches the zero-length normal.
                Matrix4 partMatrix;
                partMatrix.makeBasis(basisX, basisY, basisZ).setPosition(origin);
                Matrix3 normalMatrix;
                normalMatrix.getNormalMatrix(partMatrix);

                for (int i = 0; i < n; ++i) {
                    const float x = pa[static_cast<size_t>(i) * 3 + 0];
                    const float y = pa[static_cast<size_t>(i) * 3 + 1];
                    const float z = pa[static_cast<size_t>(i) * 3 + 2];
                    Vector3 p{origin.x + basisX.x * x + basisY.x * y + basisZ.x * z,
                              origin.y + basisX.y * x + basisY.y * y + basisZ.y * z,
                              origin.z + basisX.z * x + basisY.z * y + basisZ.z * z};
                    Vector3 nv{0.f, 1.f, 0.f};
                    if (nrm) {
                        const auto& na = nrm->array();
                        nv.set(na[static_cast<size_t>(i) * 3 + 0],
                               na[static_cast<size_t>(i) * 3 + 1],
                               na[static_cast<size_t>(i) * 3 + 2])
                                .applyMatrix3(normalMatrix);
                        if (nv.lengthSq() < 1e-12f) nv.set(0.f, 1.f, 0.f);
                        nv.normalize();
                    }
                    Vector2 t{x * texScale, y * texScale};
                    if (uv) {
                        const auto& ua = uv->array();
                        t.set(ua[static_cast<size_t>(i) * 2 + 0] * texScale,
                              ua[static_cast<size_t>(i) * 2 + 1] * texScale);
                    }
                    push(p, nv, t);
                }
                if (idx) {
                    for (int v : idx->array()) indices_.push_back(base + static_cast<unsigned int>(v));
                } else {
                    for (int i = 0; i < n; ++i) indices_.push_back(base + static_cast<unsigned int>(i));
                }
            }

            // Multiply every vertex colour by fn(x, y, z). Used to bake the
            // occlusion term once the whole building is known.
            template<class Fn>
            void modulateColors(Fn&& fn) {
                const size_t n = positions_.size() / 3;
                for (size_t i = 0; i < n; ++i) {
                    const float f = fn(positions_[i * 3 + 0], positions_[i * 3 + 1], positions_[i * 3 + 2]);
                    colors_[i * 3 + 0] *= f;
                    colors_[i * 3 + 1] *= f;
                    colors_[i * 3 + 2] *= f;
                }
            }

            [[nodiscard]] std::shared_ptr<BufferGeometry> build() const {
                auto geo = BufferGeometry::create();
                if (indices_.empty()) return geo;
                geo->setIndex(indices_);
                geo->setAttribute("position", FloatBufferAttribute::create(positions_, 3));
                geo->setAttribute("normal", FloatBufferAttribute::create(normals_, 3));
                geo->setAttribute("uv", FloatBufferAttribute::create(uvs_, 2));
                geo->setAttribute("color", FloatBufferAttribute::create(colors_, 3));
                geo->computeBoundingBox();
                geo->computeBoundingSphere();
                return geo;
            }

        private:
            void push(const Vector3& p, const Vector3& n, const Vector2& t) {
                positions_.insert(positions_.end(), {p.x, p.y, p.z});
                normals_.insert(normals_.end(), {n.x, n.y, n.z});
                uvs_.insert(uvs_.end(), {t.x, t.y});
                colors_.insert(colors_.end(), {tint.x, tint.y, tint.z});
            }

            std::vector<float> positions_, normals_, uvs_, colors_;
            std::vector<unsigned int> indices_;
        };

        // A 1-D span, used both for cutting log courses and for decomposing a
        // wall panel around its openings.
        struct Span {
            float a = 0.f, b = 0.f;
            [[nodiscard]] float length() const { return b - a; }
        };

        // span minus the union of `cuts` (which need not be sorted or disjoint).
        inline std::vector<Span> subtractSpans(Span span, std::vector<Span> cuts) {
            std::vector<Span> out;
            if (span.length() <= 0.f) return out;
            std::sort(cuts.begin(), cuts.end(), [](const Span& l, const Span& r) { return l.a < r.a; });
            float cursor = span.a;
            for (const auto& c : cuts) {
                if (c.b <= cursor) continue;
                if (c.a >= span.b) break;
                const float a = std::max(c.a, span.a);
                if (a > cursor) out.push_back({cursor, a});
                cursor = std::max(cursor, std::min(c.b, span.b));
            }
            if (cursor < span.b) out.push_back({cursor, span.b});
            return out;
        }

        // Maps a wall-local (u = along the wall, y = height, n = outward from
        // the log centreline plane) triple into world space.
        struct WallFrame {
            WallSide side = WallSide::Front;
            float halfLength = 0.f;// cabin half extent in X
            float halfDepth = 0.f; // cabin half extent in Z

            [[nodiscard]] bool alongX() const {
                return side == WallSide::Front || side == WallSide::Back;
            }
            [[nodiscard]] float outward() const {
                return (side == WallSide::Front || side == WallSide::Right) ? 1.f : -1.f;
            }
            [[nodiscard]] Vector3 point(float u, float y, float n) const {
                if (alongX()) return {u, y, outward() * (halfDepth + n)};
                return {outward() * (halfLength + n), y, u};
            }
            [[nodiscard]] Vector3 half(float hu, float hy, float hn) const {
                if (alongX()) return {hu, hy, hn};
                return {hn, hy, hu};
            }
            // Half extent of the wall along its own horizontal axis.
            [[nodiscard]] float halfSpan() const { return alongX() ? halfLength : halfDepth; }
        };

    }// namespace detail

    // ── Per-material geometry buckets ────────────────────────────────────
    struct CabinGeometry {
        std::shared_ptr<BufferGeometry> logs;    // round log courses
        std::shared_ptr<BufferGeometry> logEnds; // sawn end grain at the notches
        std::shared_ptr<BufferGeometry> shell;   // solid backing behind the logs
        std::shared_ptr<BufferGeometry> roof;    // shingled planes + ridge cap
        std::shared_ptr<BufferGeometry> trim;    // painted casings, fascia, rake
        std::shared_ptr<BufferGeometry> timber;  // deck, posts, rails, beams, door
        std::shared_ptr<BufferGeometry> glass;
        std::shared_ptr<BufferGeometry> stone;   // foundation
        std::shared_ptr<BufferGeometry> metal;   // flue, gutter, hardware
        std::shared_ptr<BufferGeometry> lamp;    // emissive sconce lenses
    };

    struct CabinMaterials {
        std::shared_ptr<MeshStandardMaterial> logs;
        std::shared_ptr<MeshStandardMaterial> logEnds;
        std::shared_ptr<MeshStandardMaterial> shell;
        std::shared_ptr<MeshStandardMaterial> roof;
        std::shared_ptr<MeshStandardMaterial> trim;
        std::shared_ptr<MeshStandardMaterial> timber;
        std::shared_ptr<MeshStandardMaterial> glass;
        std::shared_ptr<MeshStandardMaterial> stone;
        std::shared_ptr<MeshStandardMaterial> metal;
        std::shared_ptr<MeshStandardMaterial> lamp;
    };

    // ── Derived dimensions ───────────────────────────────────────────────
    inline CabinMetrics cabinMetrics(const CabinParams& p) {
        CabinMetrics m;
        const float r = p.logRadius;
        const float rise = r * (1.f - std::clamp(p.courseOverlap, 0.f, 0.45f));// vertical pitch between courses
        int courses = static_cast<int>(std::lround((p.wallHeight - 2.f * r) / rise)) + 1;
        courses = std::max(5, courses);
        // Keep the count ODD so the topmost course runs along X — the roof and
        // its rafters bear on the eave walls, not on the gable ends.
        if (courses % 2 == 0) ++courses;

        m.gradeY = 0.f;
        m.floorY = p.floorHeight;
        m.wallCourses = courses;
        m.eaveY = m.floorY + r + static_cast<float>(courses - 1) * rise + r;
        m.halfLength = p.length * 0.5f;
        m.halfDepth = p.depth * 0.5f;
        m.roofHalfLength = m.halfLength + p.rakeOverhang;
        m.roofHalfDepth = m.halfDepth + p.eaveOverhang;
        const float tanP = std::tan(p.roofPitchDeg * detail::DEG);
        m.ridgeY = m.eaveY + p.roofThickness + m.halfDepth * tanP;
        m.porchOuterZ = m.halfDepth + p.porchDepth;

        // Matches the flue construction in buildCabinGeometry: the pipe exits
        // the shingle plane at (flueX, flueZ) and rises flueRise above it, with
        // the rain cap a little higher still.
        {
            const float zc = std::clamp(p.flueZ, -m.halfDepth + 0.4f, m.halfDepth - 0.4f);
            const float xc = std::clamp(p.flueX, -m.halfLength + 0.4f, m.halfLength - 0.4f);
            const float yExit = m.ridgeY - std::abs(zc) * tanP;
            m.flueTip.set(xc, yExit + p.flueRise + 0.24f, zc);
        }
        return m;
    }

    // ── Default opening layout ───────────────────────────────────────────
    //
    // Scales with the footprint so it still reads correctly when the cabin is
    // resized: window bays are laid out on an even pitch along each wall.
    inline std::vector<Opening> defaultOpenings(const CabinParams& p) {
        const auto m = cabinMetrics(p);
        std::vector<Opening> out;

        const float winH = 1.55f;
        const float winW = 1.15f;
        const float sill = 0.95f;

        // Front (porch side): windows either side of a central door.
        {
            const float usable = p.length - 1.6f;
            const int bays = std::max(3, static_cast<int>(usable / 2.7f));
            const int doorBay = bays / 2;
            for (int i = 0; i < bays; ++i) {
                const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(bays);
                const float x = -usable * 0.5f + t * usable;
                Opening o;
                o.side = WallSide::Front;
                o.center = x;
                if (i == doorBay) {
                    o.kind = OpeningKind::Door;
                    o.width = 1.00f;
                    o.sillHeight = 0.f;
                    o.height = 2.10f;
                    o.panesX = 2;
                    o.panesY = 2;
                } else {
                    o.width = winW;
                    o.sillHeight = sill;
                    o.height = winH;
                    o.panesX = 3;
                    o.panesY = 0;
                }
                out.push_back(o);
            }
        }

        // Back: a quieter rhythm, no door.
        {
            const float usable = p.length - 2.6f;
            const int bays = std::max(2, static_cast<int>(usable / 3.6f));
            for (int i = 0; i < bays; ++i) {
                const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(bays);
                Opening o;
                o.side = WallSide::Back;
                o.center = -usable * 0.5f + t * usable;
                o.width = winW;
                o.sillHeight = sill;
                o.height = winH;
                o.panesX = 3;
                o.panesY = 0;
                out.push_back(o);
            }
        }

        // Gable ends: a pair of shuttered windows, plus one high in the gable.
        for (auto side : {WallSide::Left, WallSide::Right}) {
            const float halfGap = p.depth * 0.19f;
            for (float s : {-1.f, 1.f}) {
                Opening o;
                o.side = side;
                o.center = s * halfGap;
                o.width = 0.95f;
                o.sillHeight = sill + 0.05f;
                o.height = 1.50f;
                o.panesX = 3;
                o.panesY = 0;
                o.shutters = true;
                out.push_back(o);
            }
            Opening g;
            g.side = side;
            g.center = 0.f;
            g.width = 0.90f;
            g.sillHeight = (m.eaveY - m.floorY) + 0.55f;
            g.height = 0.95f;
            g.panesX = 2;
            g.panesY = 0;
            out.push_back(g);
        }

        return out;
    }

    // ── Geometry construction ────────────────────────────────────────────
    inline CabinGeometry buildCabinGeometry(const CabinParams& p) {

        using namespace detail;

        const CabinMetrics M = cabinMetrics(p);
        const std::vector<Opening> openings = p.openings.empty() ? defaultOpenings(p) : p.openings;

        MeshBuilder bLogs, bEnds, bShell, bRoof, bTrim, bTimber, bGlass, bStone, bMetal, bLamp;

        const float r = p.logRadius;
        const float rise = r * (1.f - std::clamp(p.courseOverlap, 0.f, 0.45f));
        const float tanP = std::tan(p.roofPitchDeg * DEG);
        const float cosP = std::cos(p.roofPitchDeg * DEG);
        const float hx = M.halfLength;
        const float hz = M.halfDepth;
        const float shellHalf = r * 0.60f;
        const float logTex = 1.f / std::max(0.2f, p.logTileLength);
        const float timberTex = 1.f / std::max(0.2f, p.timberTile);
        const float stoneTex = 1.f / std::max(0.2f, p.stoneTile);
        const float trimTex = 0.9f;

        // Roof surfaces. `roofTopY(z)` is the shingle plane; the slab underside
        // sits roofThickness below it, and meets the top plate exactly at the
        // wall line — so the gable logs can be trimmed against it.
        auto roofTopY = [&](float z) { return M.ridgeY - std::abs(z) * tanP; };
        auto roofUnderY = [&](float z) { return roofTopY(z) - p.roofThickness; };

        // ── Openings, in wall-local terms ────────────────────────────────
        struct LocalOpening {
            const Opening* src;
            float u0, u1;// along the wall
            float y0, y1;// world height
        };
        // An opening's head and sill are SNAPPED to the joint planes of the
        // wall they sit in, so a whole number of logs is removed.
        //
        // Without this the head lands mid-log, that log is cut across the full
        // opening width, and the strip between the cut and the casing shows the
        // dark backing shell — a black band above and below every window, as
        // if the logs had been sawn off short. Real log construction lands the
        // buck on a course line for exactly this reason.
        //
        // A wall's own logs occupy alternate course indices, so its joint
        // planes are at the indices of the OTHER pair of walls.
        auto snapToJoint = [&](WallSide side, float y) {
            const int wantParity = (side == WallSide::Front || side == WallSide::Back) ? 1 : 0;
            const float t = (y - M.floorY - r) / rise;
            int j = static_cast<int>(std::lround(t));
            if ((((j % 2) + 2) % 2) != wantParity) j += (t > static_cast<float>(j)) ? 1 : -1;
            return M.floorY + r + static_cast<float>(j) * rise;
        };

        std::array<std::vector<LocalOpening>, 4> byWall;
        for (const auto& o : openings) {
            LocalOpening lo{};
            lo.src = &o;
            lo.u0 = o.center - o.width * 0.5f;
            lo.u1 = o.center + o.width * 0.5f;
            const float y0 = M.floorY + o.sillHeight;
            const float y1 = y0 + o.height;
            if (o.kind == OpeningKind::Door) {
                // A door runs to the floor: its sill is the deck, not a course
                // line. Only the head needs snapping.
                lo.y0 = y0;
                lo.y1 = std::max(y0 + 4.f * rise, snapToJoint(o.side, y1));
            } else {
                // Snap the SILL, then take a whole number of course pitches for
                // the height. Snapping both edges independently rounds them
                // toward each other about half the time and loses up to a full
                // course off the opening — enough to visibly squash every
                // window on the building.
                lo.y0 = snapToJoint(o.side, y0);
                const float pitch = 2.f * rise;
                const int steps = std::max(2, static_cast<int>(std::lround(o.height / pitch)));
                lo.y1 = lo.y0 + static_cast<float>(steps) * pitch;
            }
            byWall[static_cast<size_t>(o.side)].push_back(lo);
        }
        for (auto& v : byWall) {
            std::sort(v.begin(), v.end(), [](const LocalOpening& a, const LocalOpening& b) { return a.u0 < b.u0; });
        }

        // Cuts that a log at [yLo,yHi] on `side` must respect. A window buck is
        // wider than the clear opening, so the logs are trimmed back to the
        // buck line and the casing covers the sawn ends.
        // Tested against the log's CENTRE, not its full extent. Courses
        // deliberately overlap, so every log pokes a little past the joint
        // plane its neighbour's opening was snapped to; an extent test would
        // therefore still cut the log above and below each window. What little
        // of those logs intrudes on the opening is covered by the casing.
        constexpr float BUCK = 0.045f;
        auto cutsFor = [&](WallSide side, float yCentre) {
            std::vector<Span> cuts;
            for (const auto& lo : byWall[static_cast<size_t>(side)]) {
                if (yCentre <= lo.y0 || yCentre >= lo.y1) continue;
                cuts.push_back({lo.u0 - BUCK, lo.u1 + BUCK});
            }
            return cuts;
        };

        // ── 1. Log courses ───────────────────────────────────────────────
        //
        // Courses alternate direction: even courses run along X (the eave
        // walls), odd along Z (the gable ends). Consecutive courses are one
        // radius apart vertically, so a crossing pair at a corner overlaps by
        // half a diameter — which IS a saddle notch, without any CSG. Each
        // wall's own logs then land two courses apart, i.e. exactly stacked.
        auto logTintFor = [&](int course, int wall) {
            const float h = noise::hash2(course, wall, p.seed + 601u);
            const float h2 = noise::hash2(course, wall + 17, p.seed + 607u);
            const float b = 0.84f + 0.30f * h;
            return Vector3{b * (0.98f + 0.06f * h2), b, b * (1.02f - 0.07f * h2)};
        };

        // Emit one course of logs on `side`, spanning ±halfSpan along the wall
        // plus (optionally) a notched corner overhang at each end.
        //
        // Everything about the log is perturbed per course: radius, butt-to-tip
        // taper (whose direction alternates, so successive courses lean
        // opposite ways), a mid-span bow both vertically and out of the wall
        // plane, a small offset off the wall centreline, and — most visible of
        // all — a per-END overhang length. Real notched corners are trimmed by
        // eye and never line up; a column of ends all cut to the same
        // millimetre is the giveaway that a log wall was extruded, not built.
        auto emitLogRun = [&](WallSide side, int course, float y, float halfSpan,
                              bool overhang, bool cap) {
            const WallFrame wf{side, hx, hz};
            const int wid = static_cast<int>(side) * 31;
            auto rnd = [&](int salt) { return noise::hash2(course, wid + salt, p.seed + 613u); };

            // Jitter and taper are ONE-SIDED: `logRadius` is the guaranteed
            // minimum, so however a log varies it can never drop below the
            // radius the course spacing was chosen against, and the wall can
            // never open a gap. (Symmetric ±jitter is what let daylight and
            // the grey backing shell through between courses.)
            const float rMid = r * (1.f + p.logRadiusJitter * rnd(0));
            const float taper = (rMid - r) * 0.75f * rnd(1);
            const bool buttFirst = rnd(8) > 0.5f;// which end is the butt
            const float rA = buttFirst ? rMid : rMid - taper;
            const float rB = buttFirst ? rMid - taper : rMid;
            const float rMax = std::max(rA, rB);

            // Corner ends are trimmed by eye, not sawn to a jig — but they ARE
            // trimmed. A wide spread reads as a woodpile rather than a wall;
            // keep it to about +/-12% so the column stays legible.
            const float oA = overhang ? p.cornerOverhang * (0.88f + 0.24f * rnd(2)) : 0.f;
            const float oB = overhang ? p.cornerOverhang * (0.88f + 0.24f * rnd(3)) : 0.f;
            // Kept well inside the course overlap, so wander never re-opens
            // the joint it is decorating.
            const float yOff = (rnd(4) - 0.5f) * r * 0.05f;
            const float nOff = (rnd(5) - 0.5f) * r * 0.18f;

            // Bow: a vertical sag plus an out-of-plane wander, so the wall face
            // is subtly convex/concave from log to log rather than a plane.
            Vector3 bow;
            {
                const Vector3 o = wf.point(0.f, 0.f, 1.f);
                const Vector3 c = wf.point(0.f, 0.f, 0.f);
                Vector3 outward;
                outward.subVectors(o, c);
                bow.set(0.f, p.logSag * (0.35f + 1.3f * rnd(6)), 0.f);
                bow.addScaledVector(outward, (rnd(7) - 0.5f) * r * 0.22f);
            }

            const float yc = y + yOff;
            const auto cuts = cutsFor(side, yc);
            const auto runs = subtractSpans({-halfSpan - oA, halfSpan + oB}, cuts);
            bLogs.tint = logTintFor(course, static_cast<int>(side));
            bEnds.tint = bLogs.tint;
            for (const auto& s : runs) {
                if (s.length() < rMax * 1.2f) continue;// too short to read as a log
                const Vector3 a = wf.point(s.a, yc, nOff);
                const Vector3 b = wf.point(s.b, yc, nOff);
                const float phase = noise::hash2(course, static_cast<int>(s.a * 7.f), p.seed + 619u) * 3.f;
                const float capSeed = noise::hash2(course, wid + static_cast<int>(s.a * 11.f), p.seed + 631u);
                bLogs.tube(a, b, rA, rB, p.logRadialSegments, p.logAxialSegments,
                           bow, logTex, cap ? &bEnds : nullptr, phase, capSeed);
            }
            bLogs.tint.set(1.f, 1.f, 1.f);
            bEnds.tint.set(1.f, 1.f, 1.f);
        };

        for (int k = 0; k < M.wallCourses; ++k) {
            const float y = M.floorY + r + static_cast<float>(k) * rise;
            if (k % 2 == 0) {
                emitLogRun(WallSide::Front, k, y, hx, true, true);
                emitLogRun(WallSide::Back, k, y, hx, true, true);
            } else {
                emitLogRun(WallSide::Left, k, y, hz, true, true);
                emitLogRun(WallSide::Right, k, y, hz, true, true);
            }
        }

        // ── 2. Gable log courses ─────────────────────────────────────────
        //
        // Above the top plate only the ±X walls continue, shortening as they
        // rise so their top stays under the roof underside. They carry no
        // corner overhang — the rake board covers their ends, as it does on a
        // real building.
        int gableCourse = M.wallCourses;
        {
            // Continue the odd-course (gable-wall) rhythm.
            float y = M.floorY + r + static_cast<float>(M.wallCourses) * rise;
            if (M.wallCourses % 2 == 0) y += rise;
            for (; ; y += 2.f * rise, gableCourse += 2) {
                // Half-span in Z where this log's TOP still clears the roof.
                const float avail = hz - (y + r - M.eaveY) / tanP;
                if (avail < r * 1.4f) break;
                const float span = std::min(hz, avail);
                emitLogRun(WallSide::Left, gableCourse, y, span, false, false);
                emitLogRun(WallSide::Right, gableCourse, y, span, false, false);
                if (y > M.ridgeY) break;// safety
            }
        }

        // ── 3. Solid backing shell ───────────────────────────────────────
        //
        // Round logs stacked with a chink gap are not a watertight surface —
        // between every pair of courses there is a lens-shaped hole straight
        // through the wall. A thin solid core behind them closes it, reads as
        // the chinking/interior in that gap, and costs 6 quads per panel.
        auto emitShellPanel = [&](WallSide side, float uA, float uB, float yA, float yB) {
            if (uB - uA <= 1e-4f || yB - yA <= 1e-4f) return;
            const WallFrame wf{side, hx, hz};
            bShell.box(wf.point((uA + uB) * 0.5f, (yA + yB) * 0.5f, 0.f),
                       wf.half((uB - uA) * 0.5f, (yB - yA) * 0.5f, shellHalf), 1.f);
        };

        auto emitShellWall = [&](WallSide side, float uA, float uB, float yA, float yB) {
            const auto& ops = byWall[static_cast<size_t>(side)];
            float cursor = uA;
            for (const auto& lo : ops) {
                if (lo.y1 <= yA || lo.y0 >= yB) continue;// not in this band
                const float a = std::max(uA, lo.u0 - BUCK);
                const float b = std::min(uB, lo.u1 + BUCK);
                if (b <= a) continue;
                if (a > cursor) emitShellPanel(side, cursor, a, yA, yB);
                // Below the sill and above the head.
                emitShellPanel(side, a, b, yA, std::min(yB, lo.y0));
                emitShellPanel(side, a, b, std::max(yA, lo.y1), yB);
                cursor = std::max(cursor, b);
            }
            if (cursor < uB) emitShellPanel(side, cursor, uB, yA, yB);
        };

        bShell.tint.set(1.f, 1.f, 1.f);
        emitShellWall(WallSide::Front, -hx, hx, M.floorY, M.eaveY);
        emitShellWall(WallSide::Back, -hx, hx, M.floorY, M.eaveY);
        emitShellWall(WallSide::Left, -hz, hz, M.floorY, M.eaveY);
        emitShellWall(WallSide::Right, -hz, hz, M.floorY, M.eaveY);

        // Gable triangles, as a stack of bands following the roof underside.
        // Each band is sized by the LOG clearance at its own top edge (the
        // same `+ r` rule the gable courses use), never by the bare roof line
        // — otherwise the dark shell pokes out past the stepped log ends and
        // fringes the whole rake with a black sawtooth.
        {
            const int bands = 12;
            for (auto side : {WallSide::Left, WallSide::Right}) {
                for (int i = 0; i < bands; ++i) {
                    const float y0 = M.eaveY + (M.ridgeY - p.roofThickness - M.eaveY) * static_cast<float>(i) / static_cast<float>(bands);
                    const float y1 = M.eaveY + (M.ridgeY - p.roofThickness - M.eaveY) * static_cast<float>(i + 1) / static_cast<float>(bands);
                    const float span = std::max(0.f, hz - (y1 + r * 0.5f - M.eaveY) / tanP);
                    if (span < 0.05f) continue;
                    emitShellWall(side, -span, span, y0, y1);
                }
            }
        }

        // Interior floor + ceiling, so a lit interior is never visible through
        // a window as an empty void.
        bShell.box({0.f, M.floorY - 0.05f, 0.f}, {hx, 0.05f, hz}, 1.f);

        // ── 4. Foundation ────────────────────────────────────────────────
        //
        // A rubble course is laid stone by stone, so its face is not a plane
        // and its top is not a ruled line. The body stays a single box (it is
        // never seen from inside), but the outer face is clad in short
        // segments whose depth and top height wander — enough to break the
        // hard silhouette where the stone meets the sill log.
        {
            const float fx = hx + r - p.foundationInset;
            const float fz = hz + r - p.foundationInset;
            const float top = M.floorY;
            const float bot = -p.foundationDepth;
            bStone.box({0.f, (top + bot) * 0.5f, 0.f}, {fx - 0.05f, (top - bot) * 0.5f, fz - 0.05f}, stoneTex);

            auto claddingRun = [&](bool alongX, float half, float other, float sign) {
                const float seg = 0.55f;
                const int n = std::max(2, static_cast<int>(2.f * half / seg));
                for (int i = 0; i < n; ++i) {
                    const float a = -half + 2.f * half * static_cast<float>(i) / static_cast<float>(n);
                    const float b = -half + 2.f * half * static_cast<float>(i + 1) / static_cast<float>(n);
                    const int key = i * 31 + static_cast<int>(sign) * 7 + (alongX ? 0 : 977);
                    const float bulge = 0.050f + 0.032f * noise::hash2(key, 1, p.seed + 811u);
                    const float topJit = top - 0.010f - 0.055f * noise::hash2(key, 2, p.seed + 811u);
                    const float g = 0.86f + 0.28f * noise::hash2(key, 3, p.seed + 811u);
                    bStone.tint.set(g, g * 0.995f, g * 0.985f);
                    const Vector3 c = alongX
                            ? Vector3{(a + b) * 0.5f, (topJit + bot) * 0.5f, sign * (other - 0.05f + bulge * 0.5f)}
                            : Vector3{sign * (other - 0.05f + bulge * 0.5f), (topJit + bot) * 0.5f, (a + b) * 0.5f};
                    const Vector3 h = alongX
                            ? Vector3{(b - a) * 0.5f, (topJit - bot) * 0.5f, bulge * 0.5f}
                            : Vector3{bulge * 0.5f, (topJit - bot) * 0.5f, (b - a) * 0.5f};
                    bStone.box(c, h, stoneTex, noise::hash2(key, 4, p.seed + 811u) * 3.f);
                }
            };
            claddingRun(true, fx, fz, 1.f);
            claddingRun(true, fx, fz, -1.f);
            claddingRun(false, fz, fx, 1.f);
            claddingRun(false, fz, fx, -1.f);
            bStone.tint.set(1.f, 1.f, 1.f);
        }

        // ── 5. Roof ──────────────────────────────────────────────────────
        const float rhx = M.roofHalfLength;
        const float rhz = M.roofHalfDepth;
        const float shTexU = 1.f / std::max(0.2f, p.shingleTileU);
        const float shTexV = 1.f / std::max(0.2f, p.shingleTileV);

        // v measures distance UP the slope from the eave, so shingle courses
        // stay parallel to the eave and evenly spaced regardless of pitch.
        auto roofUV = [&](float x, float z) {
            const float slope = (rhz - std::abs(z)) / cosP;
            return Vector2{x * shTexU, slope * shTexV};
        };

        for (float s : {1.f, -1.f}) {
            const std::array<Vector3, 4> top = {
                    Vector3{-rhx, roofTopY(s * rhz), s * rhz},
                    Vector3{rhx, roofTopY(s * rhz), s * rhz},
                    Vector3{rhx, M.ridgeY, 0.f},
                    Vector3{-rhx, M.ridgeY, 0.f}};
            const std::array<Vector2, 4> uv = {
                    roofUV(-rhx, s * rhz), roofUV(rhx, s * rhz),
                    roofUV(rhx, 0.f), roofUV(-rhx, 0.f)};
            // The +Z half is already CCW from above; the -Z half needs the
            // reverse order for its normal to come out up-and-outward.
            if (s > 0.f) {
                bRoof.slab(top, uv, p.roofThickness, shTexV);
            } else {
                const std::array<Vector3, 4> rev = {top[3], top[2], top[1], top[0]};
                const std::array<Vector2, 4> revUV = {uv[3], uv[2], uv[1], uv[0]};
                bRoof.slab(rev, revUV, p.roofThickness, shTexV);
            }
        }
        // Ridge cap.
        bRoof.box({0.f, M.ridgeY + 0.015f, 0.f}, {rhx, 0.055f, 0.16f}, shTexV);

        // Fascia along both eaves, rake boards up all four gable edges.
        //
        // The porch roof springs from the front eave line, so a fascia there
        // would hang straight through it — a wide white beam floating over the
        // porch. Emit the front fascia only on the stretches the porch does
        // not cover.
        {
            const float yTop = roofTopY(rhz) - p.roofThickness + 0.02f;
            // The fascia OVERLAPS the roof edge rather than butting onto it.
            // Butted flush, its inner face is exactly coplanar with the slab's
            // vertical rake/eave face and the two z-fight into a stippled line
            // along the whole roof edge.
            auto fasciaRun = [&](float s, float xA, float xB) {
                if (xB - xA < 0.05f) return;
                bTrim.box({(xA + xB) * 0.5f, yTop - p.fasciaHeight * 0.5f, s * (rhz + 0.026f)},
                          {(xB - xA) * 0.5f, p.fasciaHeight * 0.5f, 0.042f}, trimTex);
            };
            fasciaRun(-1.f, -rhx, rhx);
            if (!p.porch) {
                fasciaRun(1.f, -rhx, rhx);
            } else {
                const float px0 = std::min(p.porchStartX, p.porchEndX) - 0.20f;
                const float px1 = std::max(p.porchStartX, p.porchEndX) + 0.20f;
                fasciaRun(1.f, -rhx, std::min(px0, rhx));
                fasciaRun(1.f, std::max(px1, -rhx), rhx);
            }
        }
        for (float sx : {1.f, -1.f}) {
            for (float sz : {1.f, -1.f}) {
                // A barge board is nailed UNDER the roof deck and slightly
                // PROUD of it. Both offsets are load-bearing here: level with
                // the shingle plane its top face is coplanar with the roof, and
                // flush at the rake its outer face is coplanar with the slab's
                // side — either one z-fights along the entire gable edge.
                constexpr float rakeDrop = 0.022f;
                constexpr float rakeProud = 0.016f;
                const float x1 = sx * (rhx + rakeProud);
                const float x0 = sx * (rhx - 0.14f);
                const std::array<Vector3, 4> top = {
                        Vector3{std::min(x0, x1), roofTopY(sz * rhz) - rakeDrop, sz * rhz},
                        Vector3{std::max(x0, x1), roofTopY(sz * rhz) - rakeDrop, sz * rhz},
                        Vector3{std::max(x0, x1), M.ridgeY - rakeDrop, 0.f},
                        Vector3{std::min(x0, x1), M.ridgeY - rakeDrop, 0.f}};
                const std::array<Vector2, 4> uv = {
                        Vector2{0.f, 0.f}, Vector2{0.14f * trimTex, 0.f},
                        Vector2{0.14f * trimTex, rhz * trimTex}, Vector2{0.f, rhz * trimTex}};
                if (sz > 0.f) {
                    bTrim.slab(top, uv, p.roofThickness + 0.16f, trimTex);
                } else {
                    const std::array<Vector3, 4> rev = {top[3], top[2], top[1], top[0]};
                    const std::array<Vector2, 4> revUV = {uv[3], uv[2], uv[1], uv[0]};
                    bTrim.slab(rev, revUV, p.roofThickness + 0.16f, trimTex);
                }
            }
        }

        // ── 6. Window & door joinery ─────────────────────────────────────
        //
        // Emitted per opening in wall-local coordinates: u along the wall, n
        // outward from the log centreline plane (so n = logRadius is the log
        // face). Every part is an axis-aligned box, which keeps the winding
        // correct on all four walls without a mirrored frame.
        auto buildOpening = [&](const WallFrame& wf, const Opening& o, float u0, float u1,
                                float y0, float y1, float faceN) {
            const float w = u1 - u0;
            const float h = y1 - y0;
            const float uc = (u0 + u1) * 0.5f;
            const float yc = (y0 + y1) * 0.5f;

            // Casing (flat trim boards around the hole).
            //
            // The opening itself is cut on the JOINT between two courses, but
            // the head and sill boards are made HALF A COURSE deep so their
            // outer edges land on the mid-line of the log above and below. On
            // the joint the casing edge and the scribed groove fall on the
            // same line and merge into one ambiguous shadow; landing on the
            // belly of a log gives the trim a clean edge against solid timber,
            // which is where a real buck is fastened anyway.
            const float cw = 0.105f;                     // side casing width
            const float cvb = rise;                      // head/sill band = half a course
            const float lowBand = (o.kind == OpeningKind::Door) ? 0.f : cvb;
            const float ct = 0.030f;                     // casing half-thickness
            const float cn = faceN + 0.012f;
            const float sideHalf = (h + cvb + lowBand) * 0.5f;
            const float sideMid = (y1 + cvb + y0 - lowBand) * 0.5f;
            bTrim.box(wf.point(u0 - cw * 0.5f, sideMid, cn), wf.half(cw * 0.5f, sideHalf, ct), trimTex);
            bTrim.box(wf.point(u1 + cw * 0.5f, sideMid, cn), wf.half(cw * 0.5f, sideHalf, ct), trimTex);
            bTrim.box(wf.point(uc, y1 + cvb * 0.5f, cn), wf.half(w * 0.5f + cw, cvb * 0.5f, ct), trimTex);
            // Head drip cap, proud in Z, flush with the top of the head board
            // so the assembly still terminates exactly on the log mid-line.
            bTrim.box(wf.point(uc, y1 + cvb - 0.032f, cn + 0.030f),
                      wf.half(w * 0.5f + cw + 0.045f, 0.032f, ct + 0.038f), trimTex);

            if (o.kind == OpeningKind::Window) {
                // Sill: fills the band down to the log mid-line, proud of the
                // casing so it throws a shadow across the logs below.
                bTrim.box(wf.point(uc, y0 - cvb * 0.5f, cn + 0.030f),
                          wf.half(w * 0.5f + cw + 0.045f, cvb * 0.5f, ct + 0.042f), trimTex);
            } else {
                bTimber.box(wf.point(uc, y0 + 0.020f, cn + 0.020f),
                            wf.half(w * 0.5f + cw, 0.020f, ct + 0.055f), timberTex);
            }

            // Reveal liner: the sawn log ends inside the buck are hidden by a
            // painted jamb, exactly as a real log-wall window buck works.
            const float rn = faceN - 0.055f;
            const float rt = 0.060f;
            bTrim.box(wf.point(u0 + 0.018f, yc, rn), wf.half(0.018f, h * 0.5f, rt), trimTex);
            bTrim.box(wf.point(u1 - 0.018f, yc, rn), wf.half(0.018f, h * 0.5f, rt), trimTex);
            bTrim.box(wf.point(uc, y1 - 0.018f, rn), wf.half(w * 0.5f, 0.018f, rt), trimTex);
            bTrim.box(wf.point(uc, y0 + 0.018f, rn), wf.half(w * 0.5f, 0.018f, rt), trimTex);

            const float glassN = faceN - 0.115f;

            if (o.kind == OpeningKind::Door) {
                // Stile-and-rail door. Built as a FRAME rather than a solid
                // slab so the top lite is a genuine hole with glass behind it
                // — a slab plus a glass box at the same depth buries the glass
                // inside the timber and the lite reads as a painted panel.
                const float dn = faceN - 0.080f;// door plane
                const float dt = 0.028f;        // half-thickness of the leaf
                const float dw = w * 0.5f - 0.018f;
                const float stile = 0.115f;     // vertical edge members
                const float liteY0 = y1 - h * 0.36f;
                const float liteY1 = y1 - 0.13f;

                bTimber.tint.set(0.90f, 0.83f, 0.75f);
                // Stiles, top rail, and the rail under the lite.
                bTimber.box(wf.point(u0 + 0.018f + stile * 0.5f, (y0 + y1) * 0.5f, dn), wf.half(stile * 0.5f, h * 0.5f - 0.018f, dt), timberTex);
                bTimber.box(wf.point(u1 - 0.018f - stile * 0.5f, (y0 + y1) * 0.5f, dn), wf.half(stile * 0.5f, h * 0.5f - 0.018f, dt), timberTex);
                bTimber.box(wf.point(uc, (liteY1 + y1 - 0.018f) * 0.5f, dn), wf.half(dw, (y1 - 0.018f - liteY1) * 0.5f, dt), timberTex);
                // Everything below the lite is solid.
                bTimber.box(wf.point(uc, (y0 + 0.018f + liteY0) * 0.5f, dn), wf.half(dw, (liteY0 - y0 - 0.018f) * 0.5f, dt), timberTex);
                bTimber.tint.set(1.f, 1.f, 1.f);

                // Lite: glass sits BEHIND the leaf, muntins in front of it.
                bGlass.box(wf.point(uc, (liteY0 + liteY1) * 0.5f, dn - dt - 0.010f),
                           wf.half(dw - stile, (liteY1 - liteY0) * 0.5f, 0.010f), 1.f);
                for (int i = 1; i < std::max(1, o.panesX); ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(std::max(1, o.panesX));
                    bTrim.box(wf.point(u0 + 0.018f + (w - 0.036f) * t, (liteY0 + liteY1) * 0.5f, dn - 0.006f),
                              wf.half(0.015f, (liteY1 - liteY0) * 0.5f, 0.020f), trimTex);
                }
                for (int i = 1; i < std::max(1, o.panesY); ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(std::max(1, o.panesY));
                    bTrim.box(wf.point(uc, liteY0 + (liteY1 - liteY0) * t, dn - 0.006f),
                              wf.half(dw - stile, 0.015f, 0.020f), trimTex);
                }

                // Two raised panels below, with a clear rail between them.
                {
                    const float pTop = liteY0 - 0.075f;
                    const float pBot = y0 + 0.11f;
                    const float mid = (pTop + pBot) * 0.5f;
                    const float gap = 0.055f;
                    bTimber.box(wf.point(uc, (mid + gap + pTop) * 0.5f, dn + dt * 0.55f),
                                wf.half(dw - stile - 0.030f, (pTop - mid - gap) * 0.5f, dt * 0.55f), timberTex);
                    bTimber.box(wf.point(uc, (pBot + mid - gap) * 0.5f, dn + dt * 0.55f),
                                wf.half(dw - stile - 0.030f, (mid - gap - pBot) * 0.5f, dt * 0.55f), timberTex);
                }
                // Lever handle on a small backplate.
                bMetal.tint.set(0.34f, 0.28f, 0.22f);
                bMetal.box(wf.point(u1 - 0.150f, y0 + h * 0.45f, dn + dt + 0.006f), wf.half(0.026f, 0.055f, 0.008f), 2.f);
                bMetal.box(wf.point(u1 - 0.150f, y0 + h * 0.45f, dn + dt + 0.024f), wf.half(0.014f, 0.014f, 0.020f), 2.f);
                bMetal.box(wf.point(u1 - 0.185f, y0 + h * 0.45f, dn + dt + 0.040f), wf.half(0.048f, 0.011f, 0.011f), 2.f);
                bMetal.tint.set(1.f, 1.f, 1.f);
                return;
            }

            // ── Sash and glazing bars ─────────────────────────────────────
            //
            // A divided-light window is not one grid stretched over the hole.
            // It is one or two SASHES, each carrying its own bars, and where
            // two sashes overlap they meet at a rail that is thicker than any
            // glazing bar. Drawing a full-height grid AND a meeting rail on
            // top of it puts horizontal bars at 1/3, 1/2 and 2/3 of the
            // opening — an arrangement no real window has, and the reason the
            // pattern reads as arbitrary rather than as joinery.
            //
            // So: split tall openings into a double-hung pair, give each sash
            // its own bars, and pick the row count so the individual lights
            // come out square.
            const float sn = faceN - 0.085f;
            const float sw = 0.042f; // sash stile / rail half-width
            const float mw = 0.013f; // glazing bar half-width
            const float mr = 0.030f; // meeting rail half-height
            bTrim.box(wf.point(u0 + sw, yc, sn), wf.half(sw, h * 0.5f, 0.024f), trimTex);
            bTrim.box(wf.point(u1 - sw, yc, sn), wf.half(sw, h * 0.5f, 0.024f), trimTex);
            bTrim.box(wf.point(uc, y0 + sw, sn), wf.half(w * 0.5f, sw, 0.024f), trimTex);
            bTrim.box(wf.point(uc, y1 - sw, sn), wf.half(w * 0.5f, sw, 0.024f), trimTex);

            const float gu0 = u0 + 2.f * sw, gu1 = u1 - 2.f * sw;
            const int nx = std::max(1, o.panesX);
            const float paneW = (gu1 - gu0) / static_cast<float>(nx);

            // Short openings (gable lights, dormer) are a single casement.
            std::array<std::array<float, 2>, 2> sash{};
            int sashCount = 1;
            if (h > 1.05f) {
                bTrim.box(wf.point(uc, yc, sn), wf.half(w * 0.5f, mr, 0.026f), trimTex);
                sash[0] = {y0 + 2.f * sw, yc - mr};
                sash[1] = {yc + mr, y1 - 2.f * sw};
                sashCount = 2;
            } else {
                sash[0] = {y0 + 2.f * sw, y1 - 2.f * sw};
            }

            for (int s = 0; s < sashCount; ++s) {
                const float sa = sash[static_cast<size_t>(s)][0];
                const float sb = sash[static_cast<size_t>(s)][1];
                const float sh = sb - sa;
                if (sh <= 2.f * mw) continue;
                const int ny = (o.panesY > 0)
                        ? o.panesY
                        : std::max(1, static_cast<int>(std::lround(sh / std::max(0.02f, paneW))));
                for (int i = 1; i < nx; ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(nx);
                    bTrim.box(wf.point(gu0 + (gu1 - gu0) * t, (sa + sb) * 0.5f, sn + 0.004f),
                              wf.half(mw, sh * 0.5f, 0.018f), trimTex);
                }
                for (int i = 1; i < ny; ++i) {
                    const float t = static_cast<float>(i) / static_cast<float>(ny);
                    bTrim.box(wf.point(uc, sa + sh * t, sn + 0.004f),
                              wf.half((gu1 - gu0) * 0.5f, mw, 0.018f), trimTex);
                }
            }
            bGlass.box(wf.point(uc, yc, glassN), wf.half(w * 0.5f - sw, h * 0.5f - sw, 0.010f), 1.f);
            // Dark interior behind the glass — a window should read as depth,
            // not as a hole into the skybox.
            bShell.box(wf.point(uc, yc, glassN - 0.10f), wf.half(w * 0.5f, h * 0.5f, 0.04f), 1.f);

            if (o.shutters) {
                // A shutter pair must cover the opening when closed, so each
                // leaf is HALF the clear width — at 0.46 of the full width
                // each leaf is nearly as wide as the window and the pair reads
                // as two blank panels bolted to the wall.
                const float shw = w * 0.26f;
                // Matched to the casing assembly, not to the bare hole, so the
                // shutter tops and sills line up with the head and sill boards.
                const float shh = sideHalf;
                const float shy = sideMid;
                for (float s : {-1.f, 1.f}) {
                    const float su = (s < 0.f) ? (u0 - 0.045f - shw) : (u1 + 0.045f + shw);
                    bTrim.box(wf.point(su, shy, faceN + 0.028f), wf.half(shw, shh, 0.024f), trimTex);
                    // Board-and-batten face: vertical boards with a ledger top
                    // and bottom. A blank plank reads as a paper cut-out.
                    for (int i = 0; i < 3; ++i) {
                        const float tu = su + (static_cast<float>(i) - 1.f) * shw * 0.62f;
                        bTrim.box(wf.point(tu, shy, faceN + 0.048f), wf.half(shw * 0.26f, shh - 0.085f, 0.020f), trimTex);
                    }
                    // Inset clear of the leaf's own top/bottom faces — level
                    // with them the two coplanar surfaces stipple at grazing
                    // angles.
                    for (float sy : {-1.f, 1.f}) {
                        bTrim.box(wf.point(su, shy + sy * (shh - 0.065f), faceN + 0.050f),
                                  wf.half(shw - 0.012f, 0.040f, 0.022f), trimTex);
                    }
                }
            }
        };

        // Built from the SNAPPED extents, so the casing lands exactly on the
        // course line the logs were cut to.
        for (const auto& wallOps : byWall) {
            for (const auto& lo : wallOps) {
                const WallFrame wf{lo.src->side, hx, hz};
                buildOpening(wf, *lo.src, lo.u0, lo.u1, lo.y0, lo.y1, r);
            }
        }

        // ── 7. Wall dormer ───────────────────────────────────────────────
        //
        // A gable rising straight out of the front roof plane, its face flush
        // with the wall below. Its ridge runs in Z and dies into the main roof
        // where the two planes meet; the valley is the straight line
        // z = zRidgeBack + |x - cx| in plan, which is why the roof halves can
        // be emitted as simple planar quads.
        if (p.dormer) {
            const float cx = p.dormerCenterX;
            const float hw = p.dormerWidth * 0.5f;
            const float riseTotal = M.ridgeY - p.roofThickness - M.eaveY;
            const float yRidgeD = M.eaveY + p.roofThickness + riseTotal * std::clamp(p.dormerRiseFrac, 0.25f, 0.95f);
            const float yEaveD = yRidgeD - hw * tanP;
            // Where the dormer ridge dies into the main roof.
            const float zRidgeBack = (M.ridgeY - yRidgeD) / tanP;
            const float zFront = hz + 0.28f;// small forward projection of the rake
            const float dRake = 0.22f;
            const float faceZ = hz;

            // Face: log courses continuing the wall, clipped to the gable.
            auto dormerHalfSpanAt = [&](float y) {
                if (y <= yEaveD) return hw;
                return std::max(0.f, (yRidgeD - y) / tanP);
            };
            // Start just clear of the shingle surface at the back of a log.
            const float faceStart = roofTopY(hz - r) + r * 0.35f;
            // The dormer face has its own course grid (pitch 2*rise from
            // faceStart + r), so its window snaps to THAT, not the wall's.
            const float dCourse0 = faceStart + r;
            auto snapDormer = [&](float y) {
                const float t = (y - dCourse0) / (2.f * rise);
                return dCourse0 + (std::round(t) + 0.5f) * 2.f * rise - rise;
            };
            const float dWinY0 = snapDormer(faceStart + p.dormerWindowSill);
            const float dWinY1 = std::max(dWinY0 + 2.f * rise,
                                          snapDormer(faceStart + p.dormerWindowSill + p.dormerWindowHeight));
            for (int i = 0;; ++i) {
                const float y = faceStart + r + static_cast<float>(i) * 2.f * rise;
                // Run the courses as far into the apex as a stub of log still
                // reads; stopping early leaves a bare triangle of backing
                // under the rake boards.
                const float span = dormerHalfSpanAt(y + r);
                if (span < r * 0.75f) break;
                // Reuse the front wall's cut logic so a dormer window trims the
                // courses the same way a wall window does.
                bLogs.tint = logTintFor(1000 + i, 0);
                const WallFrame wf{WallSide::Front, hx, hz};
                std::vector<Span> cuts;
                if (y > dWinY0 && y < dWinY1)
                    cuts.push_back({cx - p.dormerWindowWidth * 0.5f - BUCK, cx + p.dormerWindowWidth * 0.5f + BUCK});
                const float rj = r * (1.f + p.logRadiusJitter * noise::hash2(i, 11, p.seed + 641u));
                const Vector3 dBow{0.f, p.logSag * 0.6f, (noise::hash2(i, 13, p.seed + 641u) - 0.5f) * r * 0.16f};
                for (const auto& s : subtractSpans({cx - span, cx + span}, cuts)) {
                    if (s.length() < r * 1.1f) continue;
                    bLogs.tube(wf.point(s.a, y, 0.f), wf.point(s.b, y, 0.f), rj, std::max(r, rj * 0.97f),
                               p.logRadialSegments, p.logAxialSegments, dBow, logTex, nullptr,
                               noise::hash2(i, 3, p.seed + 641u) * 3.f);
                }
                bLogs.tint.set(1.f, 1.f, 1.f);
            }
            // Backing shell for the face (also seals the gap down to the roof).
            // Bands only — NO flat rear panel: a full-height rectangle behind
            // the face stands proud of the dormer's own sloping roof on both
            // sides and reads as a grey billboard glued to the shingles. What
            // is actually behind the dormer is the main roof plane rising
            // away, and the window already has its own dark backing box.
            {
                // The window must be cut out of the backing, exactly as
                // emitShellWall does for the walls. Left solid, the backing
                // stands in FRONT of the recessed sash and glass — the opening
                // renders as a flat panel inside its casing with only the tips
                // of the glazing bars poking through, which is easy to mistake
                // for dark glass until the backing colour changes.
                const int bands = 12;
                const float y0all = faceStart - 0.45f;
                const float wu0 = cx - p.dormerWindowWidth * 0.5f - BUCK;
                const float wu1 = cx + p.dormerWindowWidth * 0.5f + BUCK;
                // Break the bands on the window edges so no band is half in it.
                std::vector<float> ys;
                ys.reserve(static_cast<size_t>(bands) + 3);
                for (int i = 0; i <= bands; ++i)
                    ys.push_back(y0all + (yRidgeD - y0all) * static_cast<float>(i) / static_cast<float>(bands));
                ys.push_back(dWinY0);
                ys.push_back(dWinY1);
                std::sort(ys.begin(), ys.end());
                for (size_t i = 0; i + 1 < ys.size(); ++i) {
                    const float ya = ys[i], yb = ys[i + 1];
                    if (yb - ya < 1e-3f) continue;
                    const float span = dormerHalfSpanAt(yb + r * 0.4f);
                    if (span < 0.05f) continue;
                    const float yc2 = (ya + yb) * 0.5f;
                    const float hy = (yb - ya) * 0.5f;
                    const bool inWindow = (yc2 > dWinY0 && yc2 < dWinY1);
                    if (!inWindow) {
                        bShell.box({cx, yc2, hz}, {span, hy, shellHalf}, 1.f);
                        continue;
                    }
                    for (const auto& s : subtractSpans({cx - span, cx + span}, {{wu0, wu1}})) {
                        if (s.length() < 0.02f) continue;
                        bShell.box({(s.a + s.b) * 0.5f, yc2, hz}, {s.length() * 0.5f, hy, shellHalf}, 1.f);
                    }
                }
            }

            // Skirt board along the dormer's base, hiding the shell band that
            // would otherwise show as a pale strip under the lowest log.
            bTrim.box({cx, faceStart + 0.04f, hz + r * 0.55f}, {hw, 0.075f, r * 0.55f}, trimTex);

            // Cheeks: right triangles between the main roof surface and the
            // dormer eave line.
            {
                const float zMeet = (M.ridgeY - yEaveD) / tanP;
                for (float s : {-1.f, 1.f}) {
                    const float x = cx + s * hw;
                    const Vector3 a{x, roofTopY(faceZ), faceZ};
                    const Vector3 b{x, yEaveD, faceZ};
                    const Vector3 c{x, yEaveD, std::min(zMeet, faceZ - 0.02f)};
                    bTimber.triPrism(a, b, c, {s * 0.07f, 0.f, 0.f}, timberTex);
                }
            }

            // Roof halves + rake boards.
            for (float s : {-1.f, 1.f}) {
                const float xIn = cx;
                const float xOut = cx + s * (hw + dRake);
                const float zBackIn = zRidgeBack;
                const float zBackOut = zRidgeBack + (hw + dRake);
                const float yOut = yRidgeD - (hw + dRake) * tanP;
                std::array<Vector3, 4> top = {
                        Vector3{xIn, yRidgeD, zFront},
                        Vector3{xOut, yOut, zFront},
                        Vector3{xOut, yOut, zBackOut},
                        Vector3{xIn, yRidgeD, zBackIn}};
                std::array<Vector2, 4> uv = {
                        Vector2{0.f, 0.f},
                        Vector2{(hw + dRake) / cosP * shTexV, 0.f},
                        Vector2{(hw + dRake) / cosP * shTexV, (zBackOut - zFront) * shTexU},
                        Vector2{0.f, (zBackIn - zFront) * shTexU}};
                // Make the winding come out CCW-from-above for either half.
                // The -X half traverses its plan quad clockwise as written.
                if (s < 0.f) {
                    std::swap(top[1], top[3]);
                    std::swap(uv[1], uv[3]);
                }
                bRoof.slab(top, uv, p.roofThickness * 0.85f, shTexV);

                // Rake board down the dormer's front gable edge — a sloping
                // board, not the horizontal one a single box would give.
                const float yBoardOut = yOut - 0.02f;
                const float yBoardIn = yRidgeD - 0.02f;
                std::array<Vector3, 4> rb = {
                        Vector3{xIn, yBoardIn, zFront + 0.048f},
                        Vector3{xOut, yBoardOut, zFront + 0.048f},
                        Vector3{xOut, yBoardOut, zFront - 0.028f},
                        Vector3{xIn, yBoardIn, zFront - 0.028f}};
                std::array<Vector2, 4> rbUV = {
                        Vector2{0.f, 0.f}, Vector2{(hw + dRake) / cosP * trimTex, 0.f},
                        Vector2{(hw + dRake) / cosP * trimTex, 0.065f * trimTex}, Vector2{0.f, 0.065f * trimTex}};
                if (s < 0.f) {
                    std::swap(rb[1], rb[3]);
                    std::swap(rbUV[1], rbUV[3]);
                }
                bTrim.slab(rb, rbUV, 0.17f, trimTex);
            }
            // Dormer ridge cap.
            bRoof.box({cx, yRidgeD + 0.012f, (zFront + zRidgeBack) * 0.5f},
                      {0.11f, 0.045f, (zFront - zRidgeBack) * 0.5f}, shTexV);

            // Dormer window.
            {
                Opening o;
                o.side = WallSide::Front;
                o.center = cx;
                o.width = p.dormerWindowWidth;
                o.height = dWinY1 - dWinY0;
                o.panesX = 2;
                o.panesY = 0;// derive square lights
                const WallFrame wf{WallSide::Front, hx, hz};
                buildOpening(wf, o, cx - o.width * 0.5f, cx + o.width * 0.5f, dWinY0, dWinY1, r);
            }
        }

        // ── 8. Porch ─────────────────────────────────────────────────────
        if (p.porch) {
            const float px0 = std::min(p.porchStartX, p.porchEndX);
            const float px1 = std::max(p.porchStartX, p.porchEndX);
            const float deckY = M.floorY - 0.035f;
            const float zWall = hz + r;                    // face of the logs
            const float zPost = hz + p.porchDepth;         // post centreline
            const float zDeck = zPost + 0.16f;             // deck nosing
            const float tanPP = std::tan(p.porchRoofPitchDeg * DEG);
            const float zBreak = rhz;                      // where the main eave ends
            const float yBreak = roofTopY(rhz) - p.roofThickness;
            const float zOuter = zPost + p.porchEaveOverhang;
            auto porchTopY = [&](float z) { return yBreak - (z - zBreak) * tanPP; };

            // ── Deck ──
            {
                // Rim + skirt. The rim top sits well BELOW the plank soffit so
                // the gaps between boards read as dark slots; flush with them
                // the deck renders as one unbroken sheet of wood.
                bTimber.box({(px0 + px1) * 0.5f, deckY - 0.185f, (zWall + zDeck) * 0.5f},
                            {(px1 - px0) * 0.5f, 0.10f, (zDeck - zWall) * 0.5f}, timberTex);
                bStone.box({(px0 + px1) * 0.5f, (deckY - 0.24f) * 0.5f, (zWall + zDeck) * 0.5f},
                           {(px1 - px0) * 0.5f - 0.05f, (deckY - 0.24f) * 0.5f, (zDeck - zWall) * 0.5f - 0.04f}, stoneTex);

                // Boards, laid in RUNS with staggered butt joints rather than
                // one 15 m plank per course — no sawmill sells those, and a
                // deck of unbroken full-length boards is the flattest, most
                // obviously CG surface on the whole building. Each board also
                // gets its own width, tone and a fraction of a millimetre of
                // cup, so the surface catches light unevenly.
                const float pitch = p.deckPlankWidth + p.deckPlankGap;
                const int n = std::max(1, static_cast<int>((zDeck - zWall) / pitch));
                for (int i = 0; i < n; ++i) {
                    const float zc = zWall + (static_cast<float>(i) + 0.5f) * pitch;
                    const float wJit = p.deckPlankWidth * (0.94f + 0.10f * noise::hash2(i, 3, p.seed + 709u));
                    float x = px0;
                    int seg = 0;
                    while (x < px1 - 0.02f) {
                        const float g = noise::hash2(i, 5 + seg * 7, p.seed + 701u);
                        // Board lengths ~2.5-4.6 m, staggered per course.
                        float len = 2.5f + 2.1f * noise::hash2(i * 13 + seg, 17, p.seed + 719u);
                        if (seg == 0) len *= 0.45f + 0.55f * noise::hash2(i, 23, p.seed + 727u);
                        const float xEnd = std::min(px1, x + len);
                        if (xEnd - x > 0.10f) {
                            bTimber.tint.set(0.86f + 0.26f * g, 0.87f + 0.23f * g, 0.85f + 0.27f * g);
                            bTimber.box({(x + xEnd) * 0.5f, deckY - 0.020f - 0.004f * g, zc},
                                        {(xEnd - x) * 0.5f - 0.006f, 0.020f, wJit * 0.5f},
                                        timberTex, g * 4.f, static_cast<float>(i) * 0.37f);
                        }
                        x = xEnd;
                        ++seg;
                    }
                }
                bTimber.tint.set(1.f, 1.f, 1.f);
            }

            // ── Porch roof ──
            {
                const std::array<Vector3, 4> top = {
                        Vector3{px0 - 0.20f, porchTopY(zOuter), zOuter},
                        Vector3{px1 + 0.20f, porchTopY(zOuter), zOuter},
                        Vector3{px1 + 0.20f, porchTopY(zBreak), zBreak},
                        Vector3{px0 - 0.20f, porchTopY(zBreak), zBreak}};
                const std::array<Vector2, 4> uv = {
                        Vector2{(px0 - 0.20f) * shTexU, 0.f},
                        Vector2{(px1 + 0.20f) * shTexU, 0.f},
                        Vector2{(px1 + 0.20f) * shTexU, (zOuter - zBreak) / std::cos(p.porchRoofPitchDeg * DEG) * shTexV},
                        Vector2{(px0 - 0.20f) * shTexU, (zOuter - zBreak) / std::cos(p.porchRoofPitchDeg * DEG) * shTexV}};
                bRoof.slab(top, uv, p.porchRoofThickness, shTexV);
                // Fascia + gutter along the porch eave.
                const float yEdge = porchTopY(zOuter) - p.porchRoofThickness;
                // Overlaps the slab edge — see the main fascia note above.
                bTrim.box({(px0 + px1) * 0.5f, yEdge - 0.065f, zOuter + 0.022f},
                          {(px1 - px0) * 0.5f + 0.20f, 0.075f, 0.038f}, trimTex);
                // Gutter: painted, not bright galvanised — a mirror-finish
                // downpipe reflects raw sky and reads as a chrome pole.
                bMetal.tint.set(0.42f, 0.32f, 0.24f);
                bMetal.box({(px0 + px1) * 0.5f, yEdge - 0.170f, zOuter + 0.070f},
                           {(px1 - px0) * 0.5f + 0.20f, 0.050f, 0.050f}, 1.f);
                bMetal.tint.set(1.f, 1.f, 1.f);
            }

            // ── Posts, beam, railing ──
            const float yBeamTop = porchTopY(zPost) - p.porchRoofThickness;
            const float yBeamBot = yBeamTop - p.beamHeight;
            bTimber.box({(px0 + px1) * 0.5f, (yBeamTop + yBeamBot) * 0.5f, zPost},
                        {(px1 - px0) * 0.5f, p.beamHeight * 0.5f, 0.075f}, timberTex);

            const float run = px1 - px0;
            int bays = std::max(1, static_cast<int>(std::round(run / std::max(0.8f, p.postSpacing))));
            std::vector<float> postX;
            postX.reserve(static_cast<size_t>(bays) + 1);
            for (int i = 0; i <= bays; ++i)
                postX.push_back(px0 + run * static_cast<float>(i) / static_cast<float>(bays));

            const float hp = p.postSize * 0.5f;
            for (float x : postX) {
                bTimber.box({x, (deckY + yBeamBot) * 0.5f, zPost}, {hp, (yBeamBot - deckY) * 0.5f, hp}, timberTex);
                // Cap + base blocks, so a post is not a bare stick.
                bTimber.box({x, yBeamBot - 0.045f, zPost}, {hp + 0.028f, 0.045f, hp + 0.028f}, timberTex);
                bTimber.box({x, deckY + 0.055f, zPost}, {hp + 0.028f, 0.055f, hp + 0.028f}, timberTex);
            }

            // ── Corbels (curved braces post → beam) ──
            //
            // The one place on the building where a straight edge would be
            // plainly wrong: a sawn brace has an ogee back, and nothing built
            // from boxes and cylinders can produce one. Drawn once as a 2-D
            // Shape, extruded, then folded into the timber bucket by append()
            // so it still costs no extra draw call.
            {
                // Wound COUNTER-clockwise: a clockwise outline extrudes with
                // its caps and side walls facing inward and the whole brace
                // disappears under back-face culling.
                Shape brace;
                const float bl = 0.44f;// reach along the beam
                const float bh = 0.46f;// drop down the post
                brace.moveTo(0.f, 0.f);
                brace.lineTo(0.f, -bh);
                brace.lineTo(0.055f, -bh);
                brace.quadraticCurveTo(bl * 0.30f, -bh * 0.24f, bl, 0.f);
                brace.lineTo(0.f, 0.f);

                ExtrudeGeometry::Options opt;
                opt.depth = 0.075f;
                opt.steps = 1;
                opt.curveSegments = 10;
                opt.bevelEnabled = false;
                auto braceGeo = ExtrudeGeometry::create({brace}, opt);

                for (size_t i = 0; i < postX.size(); ++i) {
                    const float x = postX[i];
                    const bool first = (i == 0), last = (i + 1 == postX.size());
                    bTimber.tint.set(0.97f, 0.97f, 0.96f);
                    if (!last) {// brace reaching in +X
                        bTimber.append(*braceGeo, {x + hp, yBeamBot, zPost - 0.0375f},
                                       {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, timberTex);
                    }
                    if (!first) {// mirrored, kept right-handed by also flipping Z
                        bTimber.append(*braceGeo, {x - hp, yBeamBot, zPost + 0.0375f},
                                       {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, -1.f}, timberTex);
                    }
                    bTimber.tint.set(1.f, 1.f, 1.f);
                }
            }

            const float stepA = p.stepsCenterX - p.stepsWidth * 0.5f;
            const float stepB = p.stepsCenterX + p.stepsWidth * 0.5f;
            const float yRailTop = deckY + p.railHeight;
            const float yRailBot = deckY + 0.16f;
            for (size_t i = 0; i + 1 < postX.size(); ++i) {
                const float a = postX[i] + hp;
                const float b = postX[i + 1] - hp;
                if (b - a < 0.15f) continue;
                // Leave the stair bay open.
                if (a < stepB && b > stepA) continue;
                const float mid = (a + b) * 0.5f;
                const float halfRun = (b - a) * 0.5f;
                bTimber.box({mid, yRailTop, zPost}, {halfRun, 0.038f, 0.075f}, timberTex);
                bTimber.box({mid, yRailTop - 0.075f, zPost}, {halfRun, 0.035f, 0.048f}, timberTex);
                bTimber.box({mid, yRailBot, zPost}, {halfRun, 0.035f, 0.048f}, timberTex);
                const int nb = std::max(1, static_cast<int>((b - a) / std::max(0.05f, p.balusterSpacing)));
                for (int k = 0; k < nb; ++k) {
                    const float t = (static_cast<float>(k) + 0.5f) / static_cast<float>(nb);
                    bTimber.box({a + (b - a) * t, (yRailTop + yRailBot) * 0.5f, zPost},
                                {p.balusterSize * 0.5f, (yRailTop - yRailBot) * 0.5f - 0.04f, p.balusterSize * 0.5f},
                                timberTex);
                }
            }

            // ── Steps ──
            //
            // Treads sit PROUD of the risers with a nosing, and the stringers
            // are thin cheeks rather than full side walls — a stair built from
            // full-depth boxes reads as a solid ramp with scored lines.
            {
                const int n = std::max(1, p.stepCount);
                const float rise = (deckY - 0.02f) / static_cast<float>(n);
                const float tread = 0.29f;
                for (int i = 0; i < n; ++i) {
                    const float yTread = rise * static_cast<float>(i + 1);
                    const float z = zDeck + tread * static_cast<float>(n - 1 - i);
                    bTimber.tint.set(0.94f + 0.10f * noise::hash2(i, 9, p.seed + 733u), 0.96f, 0.94f);
                    bTimber.box({p.stepsCenterX, yTread - 0.022f, z + tread * 0.5f + 0.025f},
                                {p.stepsWidth * 0.5f, 0.022f, tread * 0.5f + 0.025f}, timberTex);
                    bTimber.tint.set(1.f, 1.f, 1.f);
                    // Riser, set back under the nosing.
                    bTimber.box({p.stepsCenterX, yTread - rise * 0.5f - 0.022f, z + tread - 0.020f},
                                {p.stepsWidth * 0.5f - 0.02f, rise * 0.5f - 0.022f, 0.020f}, timberTex);
                }
                for (float s : {-1.f, 1.f}) {
                    bTimber.box({p.stepsCenterX + s * (p.stepsWidth * 0.5f + 0.030f), deckY * 0.5f,
                                 zDeck + tread * static_cast<float>(n) * 0.5f},
                                {0.030f, deckY * 0.5f, tread * static_cast<float>(n) * 0.5f}, timberTex);
                }
            }

            // ── Downspouts ──
            bMetal.tint.set(0.42f, 0.32f, 0.24f);
            for (float x : {postX.front(), postX.back()}) {
                const float yTop = porchTopY(zOuter) - p.porchRoofThickness - 0.20f;
                bMetal.box({x, yTop * 0.5f, zOuter + 0.070f}, {0.035f, yTop * 0.5f, 0.035f}, 1.f);
            }
            bMetal.tint.set(1.f, 1.f, 1.f);

            // ── Sconces flanking the door ──
            if (p.sconces) {
                for (const auto& o : openings) {
                    if (o.side != WallSide::Front || o.kind != OpeningKind::Door) continue;
                    const float yl = M.floorY + o.sillHeight + o.height * 0.82f;
                    for (float s : {-1.f, 1.f}) {
                        const float x = o.center + s * (o.width * 0.5f + 0.44f);
                        const float zn = hz + r;
                        bMetal.tint.set(0.30f, 0.24f, 0.19f);// dark bronze
                        bMetal.box({x, yl, zn + 0.020f}, {0.075f, 0.150f, 0.020f}, 2.f);         // backplate
                        bMetal.box({x, yl + 0.175f, zn + 0.090f}, {0.105f, 0.028f, 0.090f}, 2.f);// hood
                        bMetal.box({x, yl - 0.150f, zn + 0.080f}, {0.070f, 0.022f, 0.070f}, 2.f);// base
                        bMetal.box({x, yl + 0.215f, zn + 0.090f}, {0.022f, 0.030f, 0.022f}, 2.f);// finial
                        bMetal.tint.set(1.f, 1.f, 1.f);
                        bLamp.box({x, yl + 0.015f, zn + 0.085f}, {0.062f, 0.140f, 0.062f}, 1.f);
                    }
                }
            }
        }

        // ── 9. Flue ──────────────────────────────────────────────────────
        if (p.flue) {
            const float zc = std::clamp(p.flueZ, -hz + 0.4f, hz - 0.4f);
            const float xc = std::clamp(p.flueX, -hx + 0.4f, hx - 0.4f);
            const float yExit = roofTopY(zc);
            const float yTop = yExit + p.flueRise;
            bMetal.tint.set(1.f, 1.f, 1.f);
            const Vector3 noBow{0.f, 0.f, 0.f};
            bMetal.tube({xc, yExit - 0.5f, zc}, {xc, yTop, zc}, p.flueRadius, -1.f,
                        14, 1, noBow, 1.f, nullptr);
            // Storm collar + rain cap.
            bMetal.tube({xc, yExit + 0.03f, zc}, {xc, yExit + 0.12f, zc}, p.flueRadius * 1.45f, -1.f,
                        14, 1, noBow, 1.f, nullptr);
            bMetal.tube({xc, yTop, zc}, {xc, yTop + 0.06f, zc}, p.flueRadius * 1.7f, -1.f,
                        14, 1, noBow, 1.f, nullptr);
            bMetal.box({xc, yTop + 0.20f, zc}, {p.flueRadius * 1.9f, 0.022f, p.flueRadius * 1.9f}, 1.f);
            for (float s : {-1.f, 1.f}) {
                bMetal.box({xc + s * p.flueRadius * 1.4f, yTop + 0.11f, zc}, {0.014f, 0.090f, 0.014f}, 1.f);
            }
        }

        // ── 10. Baked ambient occlusion ──────────────────────────────────
        //
        // The single biggest thing separating this from a photograph is that a
        // covered porch renders exactly as bright as the sunlit facade. Sun
        // shadows do not fix it: under a bright sky the porch is still filled
        // by image-based light, so the deep shade that gives the building its
        // form never appears. A ray-traced backend recovers some of it, but a
        // forward GL pass has nothing at all.
        //
        // So bake the low-frequency term into the vertex colours, from the
        // geometry we already know: how deep a point sits under the porch
        // roof, how close it is under an eave, and how close it is to the
        // ground. It costs nothing at draw time and travels with the geometry
        // to any backend.
        if (p.bakeOcclusion) {
            const float px0 = std::min(p.porchStartX, p.porchEndX);
            const float px1 = std::max(p.porchStartX, p.porchEndX);
            const float deckY = M.floorY - 0.035f;
            const float zPost = hz + p.porchDepth;
            const float zOuter = zPost + p.porchEaveOverhang;
            const float zBreak = rhz;
            const float tanPP = std::tan(p.porchRoofPitchDeg * DEG);
            const float yBreak = roofTopY(rhz) - p.roofThickness;

            auto aoAt = [&](float x, float y, float z) {
                float f = 1.f;

                // Under the porch roof: darkest in the top inner corner where
                // the roof meets the wall, opening up toward the eave line.
                if (p.porch && x > px0 - 0.35f && x < px1 + 0.35f &&
                    z > -hz && z < zOuter + 0.20f && y < yBreak + 0.25f && y > deckY - 1.4f) {
                    const float depth = std::clamp((zOuter - z) / std::max(0.5f, zOuter - hz), 0.f, 1.f);
                    const float yRoof = yBreak - (std::clamp(z, zBreak, zOuter) - zBreak) * tanPP;
                    const float hgt = std::clamp((y - deckY) / std::max(0.5f, yRoof - deckY), 0.f, 1.f);
                    f *= 1.f - 0.50f * std::pow(depth, 0.75f) * (0.28f + 0.72f * hgt);
                }

                // Tucked under an eave or rake: only for points on the wall
                // envelope, so the roof planes themselves are untouched.
                if (std::abs(z) <= hz + r + 0.06f && std::abs(x) <= hx + r + 0.55f) {
                    const float d = M.eaveY - y;
                    if (d >= -0.05f && d < 1.15f)
                        f *= 1.f - 0.26f * std::clamp(1.f - d / 1.15f, 0.f, 1.f);
                }

                // Ground contact under the sill and around the foundation.
                if (y < M.floorY + 0.45f)
                    f *= 0.64f + 0.36f * std::clamp((y + 0.25f) / 1.05f, 0.f, 1.f);

                return std::clamp(f, 0.22f, 1.f);
            };

            auto bake = [&](MeshBuilder& mb) { mb.modulateColors(aoAt); };
            bake(bLogs);
            bake(bEnds);
            bake(bShell);
            bake(bTrim);
            bake(bTimber);
            bake(bStone);
            bake(bMetal);
            bake(bGlass);
            // The roof is the one surface fully open to the sky, and the lamps
            // are emissive — neither takes occlusion.
        }

        CabinGeometry g;
        g.logs = bLogs.build();
        g.logEnds = bEnds.build();
        g.shell = bShell.build();
        g.roof = bRoof.build();
        g.trim = bTrim.build();
        g.timber = bTimber.build();
        g.glass = bGlass.build();
        g.stone = bStone.build();
        g.metal = bMetal.build();
        g.lamp = bLamp.build();
        return g;
    }

    // ── Materials ────────────────────────────────────────────────────────
    inline CabinMaterials makeCabinMaterials(const CabinParams& p) {
        CabinMaterials m;
        const unsigned int S = p.textureSize;

        auto [logAlb, logNrm] = makeLogTextures(S, p.seed, p.logColor);
        auto [endAlb, endNrm] = makeLogEndTextures(S / 2, p.seed + 1u, p.logEndColor);
        auto [shAlb, shNrm] = makeShingleTextures(S, p.seed + 2u, p.shingleColor);
        auto [wdAlb, wdNrm] = makeSawnWoodTextures(S, p.seed + 3u, p.timberColor);
        auto [stAlb, stNrm] = makeStoneTextures(S, p.seed + 4u, p.stoneColor);

        auto std_ = [](Color c, float rough, float metal) {
            return MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(c).roughness(rough).metalness(metal));
        };

        m.logs = std_(Color::white, 0.88f, 0.f);
        m.logs->map = logAlb;
        m.logs->normalMap = logNrm;
        m.logs->normalScale.set(1.6f, 1.6f);
        m.logs->vertexColors = true;

        m.logEnds = std_(Color::white, 0.82f, 0.f);
        m.logEnds->map = endAlb;
        m.logEnds->normalMap = endNrm;
        m.logEnds->vertexColors = true;

        m.shell = std_(Color(p.shellColor[0], p.shellColor[1], p.shellColor[2]), 0.95f, 0.f);
        m.shell->vertexColors = true;

        m.roof = std_(Color::white, 0.94f, 0.f);
        m.roof->map = shAlb;
        m.roof->normalMap = shNrm;
        m.roof->normalScale.set(1.0f, 1.0f);
        m.roof->vertexColors = true;

        m.trim = std_(Color(p.trimColor[0], p.trimColor[1], p.trimColor[2]), 0.62f, 0.f);
        m.trim->vertexColors = true;

        m.timber = std_(Color::white, 0.85f, 0.f);
        m.timber->map = wdAlb;
        m.timber->normalMap = wdNrm;
        m.timber->normalScale.set(0.7f, 0.7f);
        m.timber->vertexColors = true;

        // Dark and reflective, so it reads as glass under IBL without needing
        // transparency (and without sorting artefacts) — but NOT optically
        // flat. A mirror-smooth pane returns the sky as one clean gradient and
        // reads as painted plastic; the ripple normal and grime roughness map
        // break the reflection up the way real cabin glazing does.
        auto [glassNrm, glassRough] = makeGlassTextures(std::max(128u, S / 2), p.seed + 5u);
        m.glass = std_(Color(0.052f, 0.062f, 0.072f), 0.42f, 0.10f);
        m.glass->normalMap = glassNrm;
        m.glass->normalScale.set(0.85f, 0.85f);
        m.glass->roughnessMap = glassRough;
        m.glass->envMapIntensity = 1.6f;
        m.glass->vertexColors = true;

        m.stone = std_(Color::white, 0.95f, 0.f);
        m.stone->map = stAlb;
        m.stone->normalMap = stNrm;
        m.stone->vertexColors = true;

        // Mostly-dielectric painted metalwork. Anything above ~0.4 metalness
        // makes the gutters and downpipes sample raw sky and read as polished
        // chrome tubing bolted to a log cabin.
        m.metal = std_(Color(0.58f, 0.585f, 0.595f), 0.52f, 0.30f);
        m.metal->vertexColors = true;

        m.lamp = std_(Color(1.f, 0.86f, 0.62f), 0.4f, 0.f);
        m.lamp->emissive = Color(1.f, 0.78f, 0.45f);
        m.lamp->emissiveIntensity = 2.2f;
        m.lamp->vertexColors = true;

        return m;
    }

    // ── Assembly ─────────────────────────────────────────────────────────
    inline std::shared_ptr<Group> createLogCabin(const CabinParams& p, const CabinMaterials& mats) {
        const CabinGeometry g = buildCabinGeometry(p);
        auto root = Group::create();
        root->name = "LogCabin";

        auto add = [&](const std::string& name,
                       const std::shared_ptr<BufferGeometry>& geo,
                       const std::shared_ptr<MeshStandardMaterial>& mat,
                       bool cast = true, bool receive = true) {
            if (!geo || !geo->hasIndex() || geo->getIndex()->count() == 0) return;
            auto mesh = Mesh::create(geo, mat);
            mesh->name = name;
            mesh->castShadow = cast;
            mesh->receiveShadow = receive;
            root->add(mesh);
        };

        add("logs", g.logs, mats.logs);
        add("logEnds", g.logEnds, mats.logEnds);
        add("shell", g.shell, mats.shell);
        add("roof", g.roof, mats.roof);
        add("trim", g.trim, mats.trim);
        add("timber", g.timber, mats.timber);
        add("glass", g.glass, mats.glass);
        add("stone", g.stone, mats.stone);
        add("metal", g.metal, mats.metal);
        add("lamp", g.lamp, mats.lamp, false, false);

        return root;
    }

    inline std::shared_ptr<Group> createLogCabin(const CabinParams& p = {}) {
        return createLogCabin(p, makeCabinMaterials(p));
    }

}// namespace threepp::architecture

#endif//THREEPP_EXTRAS_ARCHITECTURE_LOGCABIN_HPP
