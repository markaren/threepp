// Procedural low-poly bird — rest-pose template, static index buffer, baked
// countershading, and the per-frame pose bake.
//
// One bird is 94 vertices / 152 triangles. The geometry exists to carry a
// SILHOUETTE and a MOTION, not surface detail: at the 15–80 m the flock is
// meant to be seen from, the head is two pixels and the legs are one. Every
// vertex is therefore spent on wing planform, tail outline and the parts that
// move.
//
// SHADING IS PAINTED IN, NOT COMPUTED. The look comes from a per-vertex colour
// attribute written once — dark back, pale belly, dark primaries, pale
// underwing, dark tail band. Per-vertex colour multiplies albedo on BOTH
// backends (color_fragment.glsl:8; vulkan/shaders/gbuffer.frag:683), which
// makes it the one shading tool guaranteed to survive whichever renderer the
// host picks at the prompt — and it means a host scene with weak lighting still
// reads as a bird rather than a grey dart. `flatShading` has ZERO occurrences in
// the Vulkan backend, so it is never used here; the wing's hard leading- and
// trailing-edge creases come from split vertices instead.
//
// Header-only, dependency-free beyond threepp core.
//
// Coordinate convention: local +Z forward (bill), +Y up (dorsal). +X is the
// side of wing index 0; -X is wing index 1. The two wings are exact mirrors and
// every asymmetry is expressed per wing INDEX, so anatomical handedness never
// enters the code.
//
// ── Three things a reader will want to know before trusting this file ────
//
// WINDING IS VERIFIED PER TRIANGLE, NOT ASSUMED. The layout tables give ring
// traversals; whether a given traversal is counter-clockwise *as seen from
// outside* depends on the sign of the ring parameterisation, and getting it
// wrong is invisible under Side::DoubleSide and catastrophic under
// Side::FrontSide — half the bird vanishes and the other half turns inside out
// only when the camera crosses a plane. Every quad and fan below was checked by
// evaluating (b−a)×(c−a) against the outward direction at that face's own
// position, and the emitters carry the answer as a flag. Triangle COUNT and the
// per-part index RANGES are exactly as tabulated; only the corner order is the
// verified one. The mirrored wing is the interesting case: mirroring across X
// flips handedness, so wing 1's quads take the opposite corner order from wing
// 0's. The legs are NOT mirrored — both hip rings use the same ψ table and only
// the anchor's x differs — so both legs take the same order as each other.
//
// THE WING IS A CHAIN, NOT FIVE ROTATIONS ABOUT ONE PIVOT. Rotating each
// spanwise station independently about the shoulder pins every vertex to its
// rest radius and makes the distance between stations breathe by 4–8 % every
// beat: the wing pumps and fans instead of whipping. Here a running orthonormal
// frame walks outward one fixed-length segment at a time, so the wing CANNOT
// stretch whatever the constants say, and the vertex normal falls out of the
// same frame for free — which is why computeVertexNormals() is never called.
//
// THE ONE DISCONTINUOUS INPUT IS `BirdPose::feetPlanted`. Everything else in
// BirdPose enters the vertex positions continuously, so every state blend the
// caller drives (flap → glide, glide → fold, tuck → extend) is C¹ for free.
// `feetPlanted` is a bool and flipping it teleports the foot from the hanging
// position to `footWorld`. The caller MUST flip it on the frame those two
// coincide; that is the whole anti-skate contract and this file cannot enforce
// it.

#ifndef THREEPP_EXTRAS_FAUNA_BIRDGEOMETRY_HPP
#define THREEPP_EXTRAS_FAUNA_BIRDGEOMETRY_HPP

#include "threepp/core/Assert.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace threepp::fauna {

    // ── Fixed topology. These are compile-time constants, not knobs. ─────
    inline constexpr int kVertsPerBird = 94;
    inline constexpr int kTrisPerBird = 152;
    inline constexpr int kIndicesPerBird = 456;
    inline constexpr int kWingStations = 5;// per wing
    inline constexpr int kBodyRings = 4;
    inline constexpr int kBodyRadial = 5;

    // Vertex-range tags. See the layout table below for the exact index ranges
    // these name.
    enum class BirdPart : std::uint8_t {
        Body = 0,
        Head = 1,
        Wing0 = 2,
        Wing1 = 3,
        Tail = 4,
        Leg0 = 5,
        Leg1 = 6,
    };

    // ── Proportions of the unit-scale bird, metres ───────────────────────
    struct BirdShape {
        float bodyLength = 0.22f; // m, bill tip to tail tip
        float bodyRadius = 0.030f;// m, max half-width of the body spindle
        float wingSpan = 0.42f;   // m, tip to tip, fully extended
        float tailFork = 0.0f;    // -1 forked .. +1 wedge; 0 = square

        bool operator==(const BirdShape&) const = default;
    };

    // ── Plumage. LINEAR-space RGB — these multiply albedo directly. ──────
    struct BirdPlumage {
        Color back{0.125f, 0.118f, 0.112f}; // dorsal
        Color belly{0.700f, 0.680f, 0.640f};// ventral
        Color cap{0.055f, 0.055f, 0.065f};  // crown + bill
        Color leg{0.330f, 0.265f, 0.190f};
        float wingtipDark = 0.55f;    // 0..1 multiplier at the primaries
        float tailBandDark = 0.45f;   // 0..1 multiplier over the rear 25% of the tail
        float capStrength = 0.60f;    // 0..1 blend of `cap` into the crown
        float lightnessJitter = 0.06f;// ± per-bird value scatter

        bool operator==(const BirdPlumage&) const = default;
    };

    // ── The rest-pose template. Built ONCE, shared by every bird. ────────
    struct BirdTemplate {
        std::array<Vector3, kVertsPerBird> pos{};        // rest position, body space
        std::array<Vector3, kVertsPerBird> nrm{};        // rest normal, body space, unit length
        std::array<float, kVertsPerBird> span{};         // wing verts: s in 0..1 (0 shoulder, 1 tip). 0 elsewhere.
        std::array<std::uint8_t, kVertsPerBird> station{};// wing verts: 0..4. 0 elsewhere.
        std::array<BirdPart, kVertsPerBird> part{};

        Vector3 neckPivot{};               // body space
        std::array<Vector3, 2> wingRoot{}; // body space, index 0 then 1
        std::array<Vector3, 2> hipAnchor{};// body space, index 0 then 1
        Vector3 tailRoot{};                // body space
        float legLength = 0.f;             // m, hip to foot at full extension
        float tailLength = 0.f;            // m
        std::array<float, kWingStations> segSpan{};// m; segSpan[0] is unused (0)
    };

    // ── Wingbeat kinematics ──────────────────────────────────────────────
    //
    // THESE LIVE HERE BECAUSE poseBird() IS A PURE FUNCTION OF (template, pose).
    // The five-argument poseBird() below is the frozen A↔C contract and it takes
    // no parameter block, so the stroke constants cannot be read out of the
    // simulation's Params at call time. They are defaulted to exactly the values
    // the flock's own "Wingbeat kinematics" block carries; a caller that wants to
    // expose them as live knobs routes them through the six-argument overload
    // instead of editing this struct. Keeping them in one named aggregate — with
    // an operator== like every other leaf here — is what stops the two copies
    // silently drifting apart.
    struct BirdKinematics {
        float strokeDown = 0.88f;    // rad below horizontal (50°)
        float strokeUp = 0.58f;      // rad above horizontal (33°)
        float downstrokeFrac = 0.42f;// fraction of the cycle spent going down
        float spanLagCycles = 0.13f; // root→tip travelling-wave lag
        float twistAmp = 0.34f;      // rad tip twist — the scintillation
        float wristFlex = 1.00f;     // rad hand fold at mid-upstroke
        float glideDihedral = 0.10f; // rad, static dihedral held while gliding
        float perchSweep = 1.35f;    // rad, folded-wing sweep
        float perchLift = 0.20f;     // rad, folded-wing lift

        bool operator==(const BirdKinematics&) const = default;
    };

    inline constexpr BirdKinematics kDefaultKinematics{};

    // ── The A↔C contract. Everything the pose bake needs, and nothing else. ──
    // Filled by the flock's update(), consumed by poseBird(). poseBird() reads it
    // and writes vertices; it never reads or writes flock state, never allocates,
    // and never calls back.
    struct BirdPose {
        // World placement. bx/by/bz are the ORTHONORMAL body basis columns
        // (bx = the +X side, by = up, bz = forward), already carrying bank and
        // pitch. det[bx by bz] == +1 always; a mirrored basis is a bug.
        Vector3 pos{};
        Vector3 bx{1, 0, 0};
        Vector3 by{0, 1, 0};
        Vector3 bz{0, 0, 1};
        float scale = 1.f;// per-bird size multiplier, 0.90..1.10

        // Wingbeat
        float cyclePos = 0.f;  // 0..1, 0 = top of stroke
        float flapWeight = 1.f;// 0..1 amplitude envelope; 0 = glide
        float beatAmp = 1.f;   // per-bird amplitude multiplier
        float wingAsym = 0.f;  // added to wing 0's amplitude, subtracted from wing 1's
        float perchFold = 0.f; // 0..1, wings folded against the flank

        // Head (radians, about the neck pivot)
        float headYaw = 0.f;
        float headPitch = 0.f;
        float headRoll = 0.f;
        float headLead = 0.f;// m, walking head bob: +Z offset of the whole head

        // Tail
        float tailSpread = 0.10f;// 0..1
        float tailPitch = 0.f;   // rad, positive = trailing edge down
        float tailRoll = 0.f;    // rad

        // Legs. Feet are WORLD positions and are written straight through —
        // this is what stops a walking bird skating.
        std::array<Vector3, 2> footWorld{};
        float legExtend = 0.f;   // 0..1; 0 = tucked flush inside the body
        bool feetPlanted = false;// false => footWorld is ignored, legs hang from the hip

        // Body
        float bodyLift = 0.f;// m, vertical offset in the body frame (landing spring, hop arc)
    };


    namespace detail {

        // ── Layout tables. Retyping one of these wrong is the single most
        // likely defect in this file, so every one of them is asserted against a
        // geometric invariant at the bottom of makeBirdTemplate(). ───────────

        // Body: ring i, vertex j at index 5i+j. θ_j = j·72° measured from +Y.
        inline constexpr std::array<float, kBodyRings> kRingZ{-0.26f, -0.09f, 0.08f, 0.20f};// × bodyLength
        inline constexpr std::array<float, kBodyRings> kRingF{0.42f, 1.00f, 0.95f, 0.60f};  // radius factor
        inline constexpr std::array<float, kBodyRings> kRingY{-0.05f, -0.02f, 0.06f, 0.20f};// × bodyRadius

        // The body cross-section is an ellipse, taller than it is wide — a bird
        // seen head-on is a keel, not a tube.
        inline constexpr float kEllipseX = 0.90f;
        inline constexpr float kEllipseY = 1.05f;

        // Wing: station t, corner c at index 4t+c, c = {0 LEtop, 1 TEtop, 2 TEbot, 3 LEbot}.
        inline constexpr std::array<float, kWingStations> kStationS{0.00f, 0.26f, 0.52f, 0.78f, 1.00f};
        inline constexpr std::array<float, kWingStations> kStationChord{1.00f, 0.94f, 0.74f, 0.46f, 0.11f};   // × rootChord
        inline constexpr std::array<float, kWingStations> kStationZLE{0.34f, 0.40f, 0.30f, 0.02f, -0.34f};    // × rootChord
        inline constexpr std::array<float, kWingStations> kStationThick{0.055f, 0.045f, 0.030f, 0.018f, 0.006f};// × rootChord
        inline constexpr std::array<float, kWingStations> kSegSpanFrac{0.00f, 0.26f, 0.26f, 0.26f, 0.22f};    // × halfSpan

        // s^1.3, the travelling-wave lag exponent. s is a FIXED table, so this is
        // a constant — evaluating std::pow here per station per wing per bird per
        // frame is ten pow() calls a bird, about a third of the whole pose budget,
        // for a number that never changes.
        inline constexpr std::array<float, kWingStations> kSpanLagPow{0.f, 0.173566f, 0.427370f, 0.723967f, 1.f};

        inline constexpr float kRootChordFrac = 0.42f;// rootChord / halfSpan
        inline constexpr float kTailLengthFrac = 0.22f;// tailLength / bodyLength
        inline constexpr float kLegLengthFrac = 0.26f; // legLength / bodyLength

        // Tail fan half-angle at tailSpread 0 and 1.
        inline constexpr float kTailAngleMin = 0.14f;// rad
        inline constexpr float kTailAngleMax = 0.72f;// rad

        // Vertices 0..81 are posed in body space and then transformed as a block;
        // 82..93 (the legs) are written straight to world because their feet are.
        inline constexpr int kLocalVerts = 82;

        // (x, y) in units of bodyRadius against ring `i`'s elliptical shell.
        // < 1 means strictly inside. Written as a constexpr function rather than
        // a run of locals so the invariant checks below leave nothing behind in a
        // release build, where THREEPP_ASSERT does not evaluate its argument.
        [[nodiscard]] inline constexpr float shellTest(float x, float y, int i) {

            const std::size_t ii = static_cast<std::size_t>(i);
            const float u = x / (kEllipseX * kRingF[ii]);
            const float v = (y - kRingY[ii]) / (kEllipseY * kRingF[ii]);
            return u * u + v * v;
        }

        // Radius factor of the tail cone at `zFrac` · bodyLength, interpolating
        // ring 0 toward the tail apex at -0.34 L.
        [[nodiscard]] inline constexpr float tailConeFactor(float zFrac) {

            const float t = (zFrac - kRingZ[0]) / (-0.34f - kRingZ[0]);
            return kRingF[0] * (1.f - t);
        }

        [[nodiscard]] inline bool isFinite(const Vector3& v) {

            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        // frac01 handles a NEGATIVE argument, which std::fmod does not: fmod keeps
        // the sign, so a station running a lagged cycle would jump to −0.87 and
        // read the stroke backwards from a discontinuity at 0.
        [[nodiscard]] inline float frac01(float x) {

            return x - std::floor(x);
        }

        // Rodrigues on one vector. All three components are computed from the
        // ORIGINAL v; writing them back one at a time as you go silently rotates
        // about a moving target.
        inline void rodrigues(Vector3& v, const Vector3& axis, float s, float c, float k) {

            const float d = axis.dot(v);
            const float cx = axis.y * v.z - axis.z * v.y;
            const float cy = axis.z * v.x - axis.x * v.z;
            const float cz = axis.x * v.y - axis.y * v.x;

            v.set(v.x * c + cx * s + axis.x * d * k,
                  v.y * c + cy * s + axis.y * d * k,
                  v.z * c + cz * s + axis.z * d * k);
        }

        // Rotate an orthonormal frame (three column vectors) about a unit axis.
        //
        // `axis` is taken BY VALUE on purpose. The head and the wing chain both
        // rotate a frame about one of its OWN columns, and a reference parameter
        // would alias the column this call is halfway through overwriting — the
        // second and third columns would then rotate about a partially updated
        // axis. Twelve bytes of copy buys immunity from a bug that only shows up
        // as a slowly shearing wing.
        inline void rotateAboutAxis(Vector3& cx, Vector3& cy, Vector3& cz, Vector3 axis, float angle) {

            // Bit-exact identity, not an approximation: cos(0) == 1 and sin(0) == 0
            // exactly, so skipping is free of drift. Worth it — a gliding or
            // perched bird has zero deltas at most stations.
            if (angle == 0.f) return;

            const float s = std::sin(angle);
            const float c = std::cos(angle);
            const float k = 1.f - c;

            rodrigues(cx, axis, s, c, k);
            rodrigues(cy, axis, s, c, k);
            rodrigues(cz, axis, s, c, k);
        }

        // Map a body-space offset through a frame given as three columns.
        [[nodiscard]] inline Vector3 mapFrame(const Vector3& cx, const Vector3& cy, const Vector3& cz, const Vector3& o) {

            return {cx.x * o.x + cy.x * o.y + cz.x * o.z,
                    cx.y * o.x + cy.y * o.y + cz.y * o.z,
                    cx.z * o.x + cy.z * o.y + cz.z * o.z};
        }

        [[nodiscard]] inline Vector3 safeNormalized(const Vector3& v, const Vector3& fallback) {

            // Vector3::normalize() divides by length with a NaN guard but NOT a
            // zero guard, so a zero-length input yields inf/NaN and one NaN vertex
            // blows the flock's bounding sphere for the rest of the run.
            const float l2 = v.lengthSq();
            if (!(l2 > 1e-20f)) return fallback;

            Vector3 out = v;
            out.multiplyScalar(1.f / std::sqrt(l2));
            return out;
        }

        // ── The static index buffer for one bird ─────────────────────────
        //
        // Emission order is fixed so that each part occupies the index range the
        // layout table promises: body 0–119, head 120–167, wing 0 168–269,
        // wing 1 270–371, tail 372–413, leg 0 414–434, leg 1 435–455.
        [[nodiscard]] inline std::array<unsigned int, kIndicesPerBird> buildBirdIndexTemplate() {

            std::array<unsigned int, kIndicesPerBird> idx{};
            int n = 0;

            const auto tri = [&](int a, int b, int c) {
                idx[static_cast<std::size_t>(n++)] = static_cast<unsigned int>(a);
                idx[static_cast<std::size_t>(n++)] = static_cast<unsigned int>(b);
                idx[static_cast<std::size_t>(n++)] = static_cast<unsigned int>(c);
            };

            // A quad given in ring-traversal order (a, b, c, d). `mirrored` is set
            // for parts whose positions are mirrored across X — mirroring flips
            // handedness, so the same traversal produces the opposite face. Both
            // branches emit two triangles; only the corner order differs.
            const auto quad = [&](int a, int b, int c, int d, bool mirrored) {
                if (mirrored) {
                    tri(a, b, c);
                    tri(a, c, d);
                } else {
                    tri(a, c, b);
                    tri(a, d, c);
                }
            };

            // 1. Body bands — 30 tris.
            for (int i = 0; i < kBodyRings - 1; ++i) {
                for (int j = 0; j < kBodyRadial; ++j) {
                    const int j1 = (j + 1) % kBodyRadial;
                    quad(5 * i + j, 5 * i + j1, 5 * (i + 1) + j1, 5 * (i + 1) + j, false);
                }
            }
            // 2. Body tail fan — 5 tris. Apex v20 sits aft of ring 0.
            for (int j = 0; j < kBodyRadial; ++j) tri(20, j, (j + 1) % kBodyRadial);
            // 3. Body neck fan — 5 tris. Apex v21 IS the neck pivot.
            for (int j = 0; j < kBodyRadial; ++j) tri(21, 15 + (j + 1) % kBodyRadial, 15 + j);
            // 4. Head neck cap fan — 4 tris.
            for (int k = 0; k < 4; ++k) tri(22, 23 + k, 23 + (k + 1) % 4);
            // 5. Head band — 8 tris.
            for (int k = 0; k < 4; ++k) {
                const int k1 = (k + 1) % 4;
                quad(23 + k, 23 + k1, 27 + k1, 27 + k, false);
            }
            // 6. Head bill fan — 4 tris.
            for (int k = 0; k < 4; ++k) tri(31, 27 + (k + 1) % 4, 27 + k);
            // 7/8. Wings — 34 tris each. Wing 1 is the mirror, hence the flag.
            for (int w = 0; w < 2; ++w) {
                const int W = 32 + 20 * w;
                const bool mirrored = (w == 1);
                for (int t = 0; t < kWingStations - 1; ++t) {
                    const int a = W + 4 * t;
                    const int b = W + 4 * (t + 1);
                    quad(a + 0, a + 1, b + 1, b + 0, mirrored);// top sheet
                    quad(a + 1, a + 2, b + 2, b + 1, mirrored);// trailing edge
                    quad(a + 2, a + 3, b + 3, b + 2, mirrored);// bottom sheet
                    quad(a + 3, a + 0, b + 0, b + 3, mirrored);// leading edge
                }
                quad(W + 16 + 0, W + 16 + 1, W + 16 + 2, W + 16 + 3, mirrored);// tip cap
            }
            // 9. Tail upper sheet — 3 tris.
            tri(72, 74, 73);
            tri(72, 75, 74);
            tri(72, 76, 75);
            // 10. Tail lower sheet — 3 tris.
            tri(77, 78, 79);
            tri(77, 79, 80);
            tri(77, 80, 81);
            // 11. Tail rim — 8 tris. Four of the tail's five perimeter edges; the
            // fifth is the root, deliberately left open inside the body. Without
            // this rim the tail is a bare sheet that vanishes under
            // Side::FrontSide the moment the camera drops below the flock.
            quad(73, 78, 79, 74, false);
            quad(74, 79, 80, 75, false);
            quad(75, 80, 81, 76, false);
            quad(76, 81, 77, 72, false);
            // 12/13. Legs — 7 tris each. NOT mirrored: both hip rings use the same
            // ψ table and only the anchor's x differs, so the handedness is the
            // same on both sides and so is the winding.
            for (int g = 0; g < 2; ++g) {
                const int Lg = 82 + 6 * g;
                for (int m = 0; m < 3; ++m) {
                    const int m1 = (m + 1) % 3;
                    quad(Lg + m, Lg + m1, Lg + 3 + m1, Lg + 3 + m, false);
                }
                tri(Lg + 3, Lg + 5, Lg + 4);// foot cap, facing down
            }

            THREEPP_ASSERT_MSG(n == kIndicesPerBird, "bird index template is not 456 indices long");
            return idx;
        }

    }// namespace detail


    // ── Rest pose ────────────────────────────────────────────────────────
    //
    // Deterministic; no RNG. Pure function of `shape`.
    //
    // Degenerate input returns a ZERO-INITIALISED template rather than a
    // half-built one: every downstream consumer then produces a finite (if
    // invisible) bird instead of a NaN that propagates into the flock's bounding
    // sphere and, on Vulkan, into a BLAS refit.
    [[nodiscard]] inline BirdTemplate makeBirdTemplate(const BirdShape& shape) {

        BirdTemplate t{};

        const float L = shape.bodyLength;
        const float R = shape.bodyRadius;
        const float S = shape.wingSpan;

        if (!std::isfinite(L) || !std::isfinite(R) || !std::isfinite(S) || !std::isfinite(shape.tailFork)) return t;
        if (L <= 0.f || R <= 0.f || S <= 0.f) return t;

        const float H = 0.5f * S;                     // half span
        const float C0 = detail::kRootChordFrac * H;  // root chord
        const float T = detail::kTailLengthFrac * L;  // tail length
        const float legLen = detail::kLegLengthFrac * L;

        t.legLength = legLen;
        t.tailLength = T;

        // ── Body, v0..v21 ────────────────────────────────────────────────
        for (int i = 0; i < kBodyRings; ++i) {

            // Surface slope along z, by central difference where it exists. It is
            // what tilts the ring normal fore/aft — without it the body shades as
            // a cylinder and the taper toward the tail reads as a paint job.
            const int im = std::max(i - 1, 0);
            const int ip = std::min(i + 1, kBodyRings - 1);
            const float dz = (detail::kRingZ[static_cast<std::size_t>(ip)] - detail::kRingZ[static_cast<std::size_t>(im)]) * L;
            const float dr = (detail::kRingF[static_cast<std::size_t>(ip)] - detail::kRingF[static_cast<std::size_t>(im)]) * R;
            const float slope = (std::abs(dz) > 1e-9f) ? dr / dz : 0.f;

            const float f = detail::kRingF[static_cast<std::size_t>(i)];
            const float yOff = detail::kRingY[static_cast<std::size_t>(i)] * R;
            const float z = detail::kRingZ[static_cast<std::size_t>(i)] * L;

            for (int j = 0; j < kBodyRadial; ++j) {

                const float th = math::degToRad(72.f * static_cast<float>(j));
                const float sn = std::sin(th);
                const float cs = std::cos(th);

                const std::size_t v = static_cast<std::size_t>(5 * i + j);
                t.pos[v].set(f * R * detail::kEllipseX * sn,
                             yOff + f * R * detail::kEllipseY * cs,
                             z);

                Vector3 n = detail::safeNormalized({sn / detail::kEllipseX, cs / detail::kEllipseY, 0.f}, {0, 1, 0});
                n.z -= slope;
                t.nrm[v] = detail::safeNormalized(n, {0, 1, 0});
            }
        }
        t.pos[20].set(0.f, -0.03f * R, -0.34f * L);
        t.nrm[20] = detail::safeNormalized({0.f, -0.15f, -0.99f}, {0, 0, -1});
        t.pos[21].set(0.f, 0.26f * R, 0.24f * L);
        t.nrm[21] = detail::safeNormalized({0.f, 0.40f, 0.92f}, {0, 0, 1});

        t.neckPivot = t.pos[21];

        // ── Head, v22..v31 — a socket ring and a crown ring about the pivot ──
        {
            const Vector3& P = t.neckPivot;

            t.pos[22].set(P.x, P.y, P.z - 0.015f * L);
            t.nrm[22].set(0.f, 0.f, -1.f);

            for (int k = 0; k < 4; ++k) {

                const float ph = math::degToRad(90.f * static_cast<float>(k) + 45.f);
                const float sn = std::sin(ph);
                const float cs = std::cos(ph);

                const std::size_t vn = static_cast<std::size_t>(23 + k);
                t.pos[vn].set(P.x + 0.42f * R * sn, P.y + 0.42f * R * cs, P.z);
                t.nrm[vn] = detail::safeNormalized({sn, cs, -0.25f}, {0, 1, 0});

                const std::size_t vc = static_cast<std::size_t>(27 + k);
                t.pos[vc].set(P.x + 0.46f * R * sn, P.y + 0.10f * R + 0.46f * R * cs, P.z + 0.11f * L);
                t.nrm[vc] = detail::safeNormalized({sn, cs, 0.30f}, {0, 1, 0});
            }

            t.pos[31].set(P.x, P.y + 0.02f * R, P.z + 0.26f * L);
            t.nrm[31].set(0.f, 0.f, 1.f);
        }

        // ── Wings, v32..v51 (index 0) and v52..v71 (index 1) ─────────────
        //
        // Top and bottom are SPLIT VERTICES sharing a position but not a normal.
        // That is where the hard leading- and trailing-edge crease comes from, and
        // it is why `flatShading` — which does not exist on the Vulkan backend at
        // all — is never needed.
        for (int w = 0; w < 2; ++w) {

            const float sx = (w == 0) ? 1.f : -1.f;
            t.wingRoot[static_cast<std::size_t>(w)].set(sx * 0.55f * R, 0.55f * R, 0.04f * L);
            const Vector3& root = t.wingRoot[static_cast<std::size_t>(w)];

            for (int st = 0; st < kWingStations; ++st) {

                const std::size_t si = static_cast<std::size_t>(st);
                const float s = detail::kStationS[si];
                const float chord = detail::kStationChord[si] * C0;
                const float zLE = detail::kStationZLE[si] * C0;
                const float ht = detail::kStationThick[si] * C0;
                const float x = root.x + sx * s * H;

                const int base = 32 + 20 * w + 4 * st;
                const std::array<Vector3, 4> corner{
                        Vector3{x, root.y + ht, root.z + zLE},        // 0 LEtop
                        Vector3{x, root.y + ht, root.z + zLE - chord},// 1 TEtop
                        Vector3{x, root.y - ht, root.z + zLE - chord},// 2 TEbot
                        Vector3{x, root.y - ht, root.z + zLE}};       // 3 LEbot

                for (int c = 0; c < 4; ++c) {
                    const std::size_t v = static_cast<std::size_t>(base + c);
                    t.pos[v] = corner[static_cast<std::size_t>(c)];
                    t.nrm[v].set(0.f, (c <= 1) ? 1.f : -1.f, 0.f);
                    t.span[v] = s;
                    t.station[v] = static_cast<std::uint8_t>(st);
                }
            }
        }

        for (int st = 0; st < kWingStations; ++st) {
            t.segSpan[static_cast<std::size_t>(st)] = detail::kSegSpanFrac[static_cast<std::size_t>(st)] * H;
        }

        // ── Tail, v72..v81 — a slab, upper sheet then lower ──────────────
        //
        // A SLAB, NOT A SHEET: the rim quads close the trailing outline, so the
        // tail survives Side::FrontSide from below instead of disappearing every
        // time the camera drops beneath the flock. Only the ROOT edge is left
        // open, and it is the third and last of the buried holes (wing root, hip
        // ring, tail root) — the body cone is 0.315 R at z = -0.28 L against the
        // slab's 0.30 R, so the opening sits inside the body and the 4 mm slot is
        // sub-pixel long before it could be looked into.
        {
            const float ht = 0.010f * L;
            const float zRoot = -0.28f * L;
            const float zTip = zRoot - T;
            const float halfW = 0.30f * R;
            const float Wt = T * std::tan(detail::kTailAngleMin);// rest pose is spread = 0
            const float zC = zTip - shape.tailFork * 0.12f * T;

            t.tailRoot.set(0.f, 0.02f * R, zRoot);

            for (int sheet = 0; sheet < 2; ++sheet) {

                const float y = (sheet == 0) ? ht : -ht;
                const int base = 72 + 5 * sheet;

                t.pos[static_cast<std::size_t>(base + 0)].set(halfW, y, zRoot); // rootL
                t.pos[static_cast<std::size_t>(base + 1)].set(-halfW, y, zRoot);// rootR
                t.pos[static_cast<std::size_t>(base + 2)].set(-Wt, y, zTip);    // tipR
                t.pos[static_cast<std::size_t>(base + 3)].set(0.f, y, zC);      // tipC
                t.pos[static_cast<std::size_t>(base + 4)].set(Wt, y, zTip);     // tipL

                for (int k = 0; k < 5; ++k) {
                    t.nrm[static_cast<std::size_t>(base + k)].set(0.f, (sheet == 0) ? 1.f : -1.f, 0.f);
                }
            }
        }

        // ── Legs, v82..v87 (index 0) and v88..v93 (index 1) ──────────────
        //
        // No knee, deliberately. At the distances this system targets a leg is one
        // to three pixels; it exists so a perched body floats a leg-length above
        // the surface instead of melting into it, and so a hop has a visible push.
        // Between planted feet the leg simply stretches.
        for (int g = 0; g < 2; ++g) {

            const float sx = (g == 0) ? 1.f : -1.f;
            t.hipAnchor[static_cast<std::size_t>(g)].set(sx * 0.34f * R, -0.72f * R, -0.10f * L);
            const Vector3& hip = t.hipAnchor[static_cast<std::size_t>(g)];

            const int base = 82 + 6 * g;
            for (int m = 0; m < 3; ++m) {

                const float ps = math::degToRad(120.f * static_cast<float>(m));
                const float sn = std::sin(ps);
                const float cs = std::cos(ps);
                const Vector3 n = detail::safeNormalized({sn, 0.25f, cs}, {0, 1, 0});

                const std::size_t vh = static_cast<std::size_t>(base + m);
                t.pos[vh].set(hip.x + 0.110f * R * sn, hip.y, hip.z + 0.110f * R * cs);
                t.nrm[vh] = n;

                const std::size_t vf = static_cast<std::size_t>(base + 3 + m);
                t.pos[vf].set(hip.x + 0.055f * R * sn, hip.y - legLen, hip.z + 0.055f * R * cs);
                t.nrm[vf] = n;
            }
        }

        // ── Part tags ────────────────────────────────────────────────────
        for (int v = 0; v < kVertsPerBird; ++v) {
            BirdPart p = BirdPart::Body;
            if (v >= 88) p = BirdPart::Leg1;
            else if (v >= 82) p = BirdPart::Leg0;
            else if (v >= 72) p = BirdPart::Tail;
            else if (v >= 52) p = BirdPart::Wing1;
            else if (v >= 32) p = BirdPart::Wing0;
            else if (v >= 22) p = BirdPart::Head;
            t.part[static_cast<std::size_t>(v)] = p;
        }

        // ── Invariants. Cheap, and they catch the whole class of "I retyped
        // the table wrong" defect that no downstream test can localise. ──
        // Total length is exactly bodyLength: bill tip at +0.50 L, tail tip at
        // -0.50 L. Every other proportion in the tables is expressed as a fraction
        // of one of those two, so if these two hold the silhouette is the intended
        // one.
        THREEPP_ASSERT_MSG(std::abs(t.pos[31].z - 0.50f * L) < 1e-5f * L,
                           "bill apex must sit at z = +0.50 * bodyLength");
        THREEPP_ASSERT_MSG(shape.tailFork != 0.f || std::abs(t.pos[75].z + 0.50f * L) < 1e-5f * L,
                           "tail tip must sit at z = -0.50 * bodyLength when tailFork == 0");

        // Wing root and hip anchor must lie strictly INSIDE the body shell. Each
        // opens a hole in its own tube — the wing has no shoulder cap and the leg
        // has no hip cap — and those holes are invisible only because the closed
        // opaque body encloses them. Move either anchor outward and Side::FrontSide
        // starts showing the inside of the bird through the gap.
        THREEPP_ASSERT_MSG(detail::shellTest(0.55f, 0.55f, 2) < 1.f,
                           "wing root escapes the body shell at ring 2");
        THREEPP_ASSERT_MSG(detail::shellTest(0.34f, -0.72f, 1) < 1.f,
                           "hip anchor escapes the body shell at ring 1");

        // The tail slab's root edge must meet a body at least as tall as itself.
        // (In x the cone is 0.90 × that, so the root corners sit a fraction of a
        // millimetre proud — the slab is opaque and it does not read.)
        THREEPP_ASSERT_MSG(0.30f < detail::tailConeFactor(-0.28f) * detail::kEllipseY,
                           "tail root is wider than the body cone at z = -0.28 * bodyLength");

        THREEPP_ASSERT_MSG(std::abs(t.segSpan[1] + t.segSpan[2] + t.segSpan[3] + t.segSpan[4] - H) < 1e-5f * H,
                           "wing segment spans must sum to the half span");

        return t;
    }


    // ── Index buffer ─────────────────────────────────────────────────────
    //
    // Bird b occupies [b*kIndicesPerBird, (b+1)*kIndicesPerBird) with every value
    // offset by b*kVertsPerBird. Emitted as `unsigned int` to match
    // IntBufferAttribute; hand the result to BufferGeometry::setIndex through the
    // move overload so a 256-bird buffer is not copied on the way in.
    [[nodiscard]] inline std::vector<unsigned int> makeFlockIndices(int birdCount) {

        std::vector<unsigned int> out;
        if (birdCount <= 0) return out;

        const auto base = detail::buildBirdIndexTemplate();

        out.resize(static_cast<std::size_t>(birdCount) * kIndicesPerBird);
        for (int b = 0; b < birdCount; ++b) {
            const unsigned int vOff = static_cast<unsigned int>(b) * kVertsPerBird;
            const std::size_t iOff = static_cast<std::size_t>(b) * kIndicesPerBird;
            for (std::size_t k = 0; k < static_cast<std::size_t>(kIndicesPerBird); ++k) {
                out[iOff + k] = base[k] + vOff;
            }
        }

        THREEPP_ASSERT_MSG(out.size() == static_cast<std::size_t>(birdCount) * kIndicesPerBird,
                           "flock index buffer length mismatch");
        THREEPP_ASSERT_MSG(*std::max_element(out.begin(), out.end()) ==
                                   static_cast<unsigned int>(birdCount * kVertsPerBird - 1),
                           "flock index buffer does not reference the last vertex");
        return out;
    }


    // ── Colour bake ──────────────────────────────────────────────────────
    //
    // Written once at construction, never touched again. `seed` drives the
    // per-bird lightness jitter ONLY — the per-vertex pattern is identical on
    // every bird, so it is computed once and then scaled.
    //
    // THE DARK WINGTIP OVER A PALE UNDERWING IS THE HIGHEST-VALUE LINE IN HERE.
    // It is what the eye tracks through the beat: the underside flashes pale on
    // the upstroke and the tip stays dark, which is the whole reason a distant
    // flock reads as birds and not as drifting confetti. Without it a uniformly
    // pale wing reads as paper.
    [[nodiscard]] inline std::vector<float> makeFlockColors(const BirdTemplate& tmpl,
                                                            const BirdPlumage& plumage,
                                                            int birdCount,
                                                            unsigned int seed) {

        std::vector<float> out;
        if (birdCount <= 0) return out;

        std::array<Color, kVertsPerBird> base{};

        for (int v = 0; v < kVertsPerBird; ++v) {

            const std::size_t vi = static_cast<std::size_t>(v);
            const BirdPart p = tmpl.part[vi];

            // Countershading: dark where the surface looks up, pale where it looks
            // down. The -0.6 lower edge, rather than -1, keeps the flanks from
            // going fully belly-pale and losing the body's roundness.
            float shade = math::smoothstep(-0.6f, 1.0f, tmpl.nrm[vi].y);
            float wash = 1.f;

            if (p == BirdPart::Wing0 || p == BirdPart::Wing1) {

                // Corner index within the station: 0/1 are the top sheet, 2/3 the
                // bottom. A wing sheet is flat, so its geometric normal carries no
                // countershading information at all — the two sheets have to be
                // painted, not shaded.
                const int c = (v - ((p == BirdPart::Wing0) ? 32 : 52)) % 4;
                shade = (c <= 1) ? 1.00f : 0.15f;
                const float s = tmpl.span[vi];
                wash = math::lerp(1.f, plumage.wingtipDark, s * s);

            } else if (p == BirdPart::Tail) {

                shade = (v <= 76) ? 1.00f : 0.15f;
                const bool tip = (v >= 74 && v <= 76) || (v >= 79 && v <= 81);
                if (tip) wash = plumage.tailBandDark;
            }

            Color col;
            col.lerpColors(plumage.belly, plumage.back, shade);
            col.multiplyScalar(wash);

            if (p == BirdPart::Head && v >= 27) col.lerp(plumage.cap, plumage.capStrength);
            if (p == BirdPart::Leg0 || p == BirdPart::Leg1) col = plumage.leg;

            base[vi] = col;
        }

        // One draw per bird, hoisted into its own named const — two draws inside a
        // single expression would make "deterministic for a fixed seed" depend on
        // the compiler's argument evaluation order. math::Rng carries the
        // explicit [0,1) conversion (uniform_real_distribution's mapping is
        // implementation-defined).
        math::Rng rng(seed ? seed : 1u);
        const float jitterRange = std::max(plumage.lightnessJitter, 0.f);

        out.resize(static_cast<std::size_t>(birdCount) * kVertsPerBird * 3u);

        for (int b = 0; b < birdCount; ++b) {

            const float u = rng.nextFloat();
            const float jitter = (u * 2.f - 1.f) * jitterRange;
            const float k = 1.f + jitter;

            std::size_t o = static_cast<std::size_t>(b) * kVertsPerBird * 3u;
            for (int v = 0; v < kVertsPerBird; ++v) {
                const Color& c = base[static_cast<std::size_t>(v)];
                out[o++] = std::clamp(c.r * k, 0.f, 1.f);
                out[o++] = std::clamp(c.g * k, 0.f, 1.f);
                out[o++] = std::clamp(c.b * k, 0.f, 1.f);
            }
        }

        return out;
    }


    // ── The per-frame pose bake ──────────────────────────────────────────
    //
    // Writes exactly kVertsPerBird positions and kVertsPerBird normals in WORLD
    // space into posOut/nrmOut at [vertBase*3, (vertBase+94)*3).
    // Allocation-free. Deterministic. Called once per bird per baked frame.
    //
    // Three passes: (1) pose vertices 0..81 in body space, (2) transform them to
    // world through the body basis, (3) write the legs 82..93 straight to world,
    // because their feet already are.
    inline void poseBird(const BirdTemplate& tmpl,
                         const BirdPose& pose,
                         const BirdKinematics& kin,
                         std::vector<float>& posOut,
                         std::vector<float>& nrmOut,
                         int vertBase) {

        if (vertBase < 0) return;

        const std::size_t need = (static_cast<std::size_t>(vertBase) + kVertsPerBird) * 3u;
        if (posOut.size() < need || nrmOut.size() < need) return;

        // A non-finite pose writes 94 non-finite vertices, and ONE of those blows
        // the flock's bounding sphere for the rest of the session (and can wreck a
        // Vulkan BLAS refit). Leaving last frame's vertices in place is the
        // harmless failure: the bird freezes, nothing else notices.
        if (!detail::isFinite(pose.pos) || !std::isfinite(pose.scale) ||
            !detail::isFinite(pose.bx) || !detail::isFinite(pose.by) || !detail::isFinite(pose.bz)) return;

        std::array<Vector3, detail::kLocalVerts> lp{};
        std::array<Vector3, detail::kLocalVerts> ln{};

        // bodyLift is a WHOLE-BODY offset, not a body-rings-only one: the landing
        // spring compresses the bird toward its planted feet. Applying it to
        // vertices 0..21 alone shears the head off the neck by up to a neck radius
        // at the exact moment the eye is watching the touchdown.
        const float lift = pose.bodyLift;
        const float flapWeight = std::clamp(pose.flapWeight, 0.f, 1.f);
        const float perchFold = std::clamp(pose.perchFold, 0.f, 1.f);

        // ── Pass 1a: body, v0..v21 ───────────────────────────────────────
        for (int v = 0; v <= 21; ++v) {
            const std::size_t vi = static_cast<std::size_t>(v);
            lp[vi] = tmpl.pos[vi];
            lp[vi].y += lift;
            ln[vi] = tmpl.nrm[vi];
        }

        // ── Pass 1b: head, v22..v31, about the neck pivot ────────────────
        {
            Vector3 Hx{1, 0, 0}, Hy{0, 1, 0}, Hz{0, 0, 1};
            detail::rotateAboutAxis(Hx, Hy, Hz, Vector3{0, 1, 0}, pose.headYaw);
            detail::rotateAboutAxis(Hx, Hy, Hz, Hx, pose.headPitch);
            detail::rotateAboutAxis(Hx, Hy, Hz, Hz, pose.headRoll);

            for (int v = 22; v <= 31; ++v) {
                const std::size_t vi = static_cast<std::size_t>(v);
                const Vector3 o = tmpl.pos[vi] - tmpl.neckPivot;
                lp[vi] = tmpl.neckPivot + detail::mapFrame(Hx, Hy, Hz, o);
                lp[vi].y += lift;
                lp[vi].z += pose.headLead;
                ln[vi] = detail::mapFrame(Hx, Hy, Hz, tmpl.nrm[vi]);
            }
        }

        // ── Pass 1c: wings, v32..v71 ─────────────────────────────────────
        for (int w = 0; w < 2; ++w) {

            const float sx = (w == 0) ? 1.f : -1.f;
            const float asym = (w == 0) ? pose.wingAsym : -pose.wingAsym;
            const float A = std::clamp(flapWeight * pose.beatAmp * (1.f - perchFold) + asym, 0.f, 1.6f);
            const float kD = std::clamp(kin.downstrokeFrac, 0.05f, 0.95f);

            std::array<float, kWingStations> theta{}, twist{}, sweepExtra{}, liftExtra{}, spanScale{};

            for (int t = 0; t < kWingStations; ++t) {

                const std::size_t ti = static_cast<std::size_t>(t);
                const std::size_t v0 = static_cast<std::size_t>(32 + 20 * w + 4 * t);
                const float s = tmpl.span[v0];

                // SPANWISE TRAVELLING WAVE. The tip reaches its extreme about an
                // eighth of a cycle after the root, exactly as a flexible
                // cantilever does. One subtraction, and it is the difference
                // between a wing and a plank.
                const float p_t = detail::frac01(pose.cyclePos - kin.spanLagCycles * detail::kSpanLagPow[ti]);

                // DUTY WARP: the downstroke is the power stroke. A raw cosine gives
                // a symmetric wing-wiper; a real wing snaps down and eases up.
                //
                // The warp's breakpoint lands at q = 0.5 — the bottom of the
                // stroke — where sin(2πq) is exactly zero. That is what keeps the
                // DIHEDRAL C¹ across it: dk/dp carries a factor of sin(2πq) and so
                // vanishes from both sides, no matter how lopsided dq/dp is. The
                // elevation, which is the whole silhouette at range, therefore has
                // no kink. Twist and wrist fold do have one there and at the 0/1
                // wrap — see the fold below; that break is deliberate and it is
                // where a real wrist actually snaps.
                const float q_t = (p_t < kD) ? 0.5f * p_t / kD
                                             : 0.5f + 0.5f * (p_t - kD) / (1.f - kD);

                const float k_t = std::cos(math::TWO_PI * q_t);// +1 top, -1 bottom
                const float sn = std::sin(math::TWO_PI * q_t);

                // ARC ASYMMETRY, C¹. The wing goes ~50° below horizontal and only
                // ~33° above. Blending the two MAGNITUDES by (0.5 + 0.5k) and
                // multiplying by k keeps the derivative continuous at the
                // horizontal crossing; a `k > 0 ? a : b` branch steps dθ/dk by 50 %
                // exactly where the wing is fastest, twice a beat, which the eye
                // reads as a mechanical tick.
                const float ampMag = math::lerp(kin.strokeDown, kin.strokeUp, 0.5f + 0.5f * k_t);

                theta[ti] = A * ampMag * k_t * (0.55f + 0.45f * s) +
                            (1.f - flapWeight) * kin.glideDihedral * (0.3f + 0.7f * s) +
                            perchFold * kin.perchLift * s;

                // TWIST IS IN QUADRATURE WITH DIHEDRAL, NOT IN PHASE. Feathering
                // has to peak at mid-stroke where the wing is fastest and pass
                // through zero at the top and bottom where it is momentarily
                // stationary. In phase, the wing pronates when it should be neutral
                // and the whole cue collapses into an unmotivated roll wobble.
                twist[ti] = A * kin.twistAmp * s * s * sn;// + = leading edge down

                // WRIST FOLD, AND IT MUST BE ON THE UPSTROKE HALF. max(0, -sin) is
                // exactly zero through the entire downstroke and peaks at
                // mid-upstroke. This term alone produces the tip's closed loop
                // (outer arc down at full span, inner arc up at reduced span),
                // which is why there is no separate sweep term: a sweep in phase
                // with elevation traces a line, not an ellipse.
                //
                // The max() puts a slope break in the fold RATE at the bottom of
                // the stroke and again at the 0/1 wrap: position and velocity stay
                // finite (measured max |dp/dcyclePos| ≈ 0.93 m per cycle) but
                // acceleration jumps. That is the one place in this file where the
                // pose is C⁰-with-bounded-derivative rather than C¹, it is
                // deliberate, and it is at the exact phase where the dihedral rate
                // is zero — so the hand starts folding out of a stationary wing,
                // which is what a wrist snap looks like. Every STATE blend the
                // caller drives (flapWeight, perchFold, tailSpread, legExtend)
                // stays C¹; do not confuse the two when reading §4's C¹ claim.
                const float u = std::max(0.f, -sn);
                const float fold = A * kin.wristFlex * u * ((s >= 0.5f) ? 1.f : 0.25f);

                sweepExtra[ti] = fold + perchFold * kin.perchSweep * s;
                liftExtra[ti] = 0.35f * fold;
                spanScale[ti] = 1.f - 0.45f * perchFold * ((t >= 2) ? 1.f : 0.33f);
            }

            // A RUNNING FRAME, ONE FIXED-LENGTH SEGMENT AT A TIME. Fixed segment
            // lengths mean the wing cannot stretch whatever the constants say, and
            // the normal falls out of the same orthonormal frame.
            //
            // The fold shortens the PROJECTED span from 0.445 m to 0.263 m over a
            // beat — a 41 % pulse, measured, not estimated. Past ~40 m the wing's
            // shape is no longer resolvable but its horizontal extent still is, and
            // a non-pulsing extent reads as a flapping cross-shaped kite. Do not
            // remove this term to save a max().
            Vector3 Fx{sx, 0, 0}, Fy{0, 1, 0}, Fz{0, 0, 1};
            Vector3 P = tmpl.wingRoot[static_cast<std::size_t>(w)];

            for (int t = 0; t < kWingStations; ++t) {

                const std::size_t ti = static_cast<std::size_t>(t);
                const float dTheta = theta[ti] - (t ? theta[ti - 1] : 0.f);
                const float dTwist = twist[ti] - (t ? twist[ti - 1] : 0.f);
                const float dSweep = sweepExtra[ti] - (t ? sweepExtra[ti - 1] : 0.f);
                const float dLift = liftExtra[ti] - (t ? liftExtra[ti - 1] : 0.f);

                // ROTATION ORDER IS FIXED: dihedral+lift about Fz, sweep about Fy,
                // twist about Fx — all applied to the RUNNING frame, in that order.
                // Signs are per wing INDEX via sx, so the two wings are exact
                // mirrors and anatomical handedness never enters the code. The
                // sweep sign is the one worth checking by hand: for wing 0 a
                // positive dSweep must move Fx toward -Z (aft), which is what
                // rotating the frame about +Y by +dSweep does.
                detail::rotateAboutAxis(Fx, Fy, Fz, Fz, sx * (dTheta + dLift));
                detail::rotateAboutAxis(Fx, Fy, Fz, Fy, sx * dSweep);
                detail::rotateAboutAxis(Fx, Fy, Fz, Fx, sx * dTwist);

                if (t > 0) P.addScaledVector(Fx, tmpl.segSpan[ti] * spanScale[ti]);

                // The station's own origin: the span point on the wing axis. All
                // four corners of a station share it, so reading its x off any of
                // them keeps this independent of halfSpan.
                const std::size_t v0 = static_cast<std::size_t>(32 + 20 * w + 4 * t);
                const Vector3 origin{tmpl.pos[v0].x,
                                     tmpl.wingRoot[static_cast<std::size_t>(w)].y,
                                     tmpl.wingRoot[static_cast<std::size_t>(w)].z};

                for (int c = 0; c < 4; ++c) {
                    const std::size_t vi = v0 + static_cast<std::size_t>(c);
                    const Vector3 o = tmpl.pos[vi] - origin;
                    lp[vi] = P + detail::mapFrame(Fx, Fy, Fz, o);
                    lp[vi].y += lift;
                    ln[vi] = detail::mapFrame(Fx, Fy, Fz, tmpl.nrm[vi]);
                }
            }
        }

        // ── Pass 1d: tail, v72..v81 ──────────────────────────────────────
        {
            const float spread = std::clamp(pose.tailSpread, 0.f, 1.f);
            const float halfAngle = math::lerp(detail::kTailAngleMin, detail::kTailAngleMax, spread);
            const float Wt = tmpl.tailLength * std::tan(halfAngle);

            Vector3 Tx{1, 0, 0}, Ty{0, 1, 0}, Tz{0, 0, 1};
            detail::rotateAboutAxis(Tx, Ty, Tz, Vector3{1, 0, 0}, -pose.tailPitch);
            detail::rotateAboutAxis(Tx, Ty, Tz, Vector3{0, 0, 1}, pose.tailRoll);

            for (int v = 72; v <= 81; ++v) {

                const std::size_t vi = static_cast<std::size_t>(v);
                Vector3 r = tmpl.pos[vi];

                // Fanning the tail moves only the two outer tip vertices of each
                // sheet; the centre tip carries the fork and does not spread.
                if (v == 74 || v == 79) r.x = -Wt;
                else if (v == 76 || v == 81) r.x = Wt;

                const Vector3 o = r - tmpl.tailRoot;
                lp[vi] = tmpl.tailRoot + detail::mapFrame(Tx, Ty, Tz, o);
                lp[vi].y += lift;
                ln[vi] = detail::mapFrame(Tx, Ty, Tz, tmpl.nrm[vi]);
            }
        }

        // ── Pass 2: body space → world, v0..v81 ──────────────────────────
        //
        // The result is already unit length — the basis is orthonormal and the
        // local normals are unit. DO NOT renormalise: it is a wasted sqrt per
        // vertex and, worse, a zero-length input would turn a harmless stale
        // normal into a NaN.
        {
            const Vector3& bx = pose.bx;
            const Vector3& by = pose.by;
            const Vector3& bz = pose.bz;
            const float sc = pose.scale;

            for (int v = 0; v < detail::kLocalVerts; ++v) {

                const std::size_t vi = static_cast<std::size_t>(v);
                const Vector3& l = lp[vi];
                const Vector3& n = ln[vi];
                const std::size_t o = (static_cast<std::size_t>(vertBase) + vi) * 3u;

                posOut[o + 0] = pose.pos.x + (bx.x * l.x + by.x * l.y + bz.x * l.z) * sc;
                posOut[o + 1] = pose.pos.y + (bx.y * l.x + by.y * l.y + bz.y * l.z) * sc;
                posOut[o + 2] = pose.pos.z + (bx.z * l.x + by.z * l.y + bz.z * l.z) * sc;

                nrmOut[o + 0] = bx.x * n.x + by.x * n.y + bz.x * n.z;
                nrmOut[o + 1] = bx.y * n.x + by.y * n.y + bz.y * n.z;
                nrmOut[o + 2] = bx.z * n.x + by.z * n.y + bz.z * n.z;
            }
        }

        // ── Pass 3: legs, v82..v93, written straight to world ────────────
        //
        // IN FLIGHT legExtend is 0 and feetPlanted is false, which puts the whole
        // leg inside the closed opaque body: invisible, and never degenerate.
        // There is no collapsed vertex anywhere in this system, which is why the
        // zero-length-normal question never arises.
        {
            const float sc = pose.scale;
            const float extend = std::clamp(pose.legExtend, 0.f, 1.f);

            for (int g = 0; g < 2; ++g) {

                const std::size_t gi = static_cast<std::size_t>(g);

                Vector3 hipLocal = tmpl.hipAnchor[gi];
                hipLocal.y += lift;
                Vector3 hipW = pose.pos + detail::mapFrame(pose.bx, pose.by, pose.bz, hipLocal) * sc;

                Vector3 footW;
                const bool planted = pose.feetPlanted && detail::isFinite(pose.footWorld[gi]);
                if (planted) {
                    footW = pose.footWorld[gi];
                } else {
                    const float len = tmpl.legLength * sc * math::lerp(0.15f, 1.f, extend);
                    footW = hipW;
                    footW.addScaledVector(pose.by, -len);
                }

                // Ring orientation from the hip→foot axis. Every normalisation here
                // has an explicit fallback: a bird whose foot lands exactly on its
                // hip is rare, and a NaN leg is forever.
                Vector3 axis = footW - hipW;
                {
                    Vector3 down = pose.by;
                    down.negate();
                    axis = detail::safeNormalized(axis, down);
                }
                Vector3 side = pose.bx;
                side.addScaledVector(axis, -side.dot(axis));
                side = detail::safeNormalized(side, pose.bz);

                Vector3 fwd;
                fwd.crossVectors(axis, side);

                // (side, up, fwd) is the rest frame's (X, Y, Z) carried onto the
                // posed leg, so rest offsets and rest normals map through unchanged.
                Vector3 up = axis;
                up.negate();

                const int base = 82 + 6 * g;
                const Vector3& hipRest = tmpl.hipAnchor[gi];
                Vector3 footRest = hipRest;
                footRest.y -= tmpl.legLength;

                for (int m = 0; m < 3; ++m) {
                    for (int ring = 0; ring < 2; ++ring) {

                        const std::size_t vi = static_cast<std::size_t>(base + 3 * ring + m);
                        const Vector3& anchorRest = (ring == 0) ? hipRest : footRest;
                        const Vector3& anchorW = (ring == 0) ? hipW : footW;

                        const Vector3 o = tmpl.pos[vi] - anchorRest;
                        const Vector3 r = detail::mapFrame(side, up, fwd, o);
                        const Vector3 n = detail::mapFrame(side, up, fwd, tmpl.nrm[vi]);

                        const std::size_t io = (static_cast<std::size_t>(vertBase) + vi) * 3u;
                        posOut[io + 0] = anchorW.x + r.x * sc;
                        posOut[io + 1] = anchorW.y + r.y * sc;
                        posOut[io + 2] = anchorW.z + r.z * sc;

                        nrmOut[io + 0] = n.x;
                        nrmOut[io + 1] = n.y;
                        nrmOut[io + 2] = n.z;
                    }
                }
            }
        }
    }

    // The frozen five-argument form: identical behaviour with the stock stroke
    // constants. Callers that expose the kinematics as live knobs use the
    // overload above.
    inline void poseBird(const BirdTemplate& tmpl,
                         const BirdPose& pose,
                         std::vector<float>& posOut,
                         std::vector<float>& nrmOut,
                         int vertBase) {

        poseBird(tmpl, pose, kDefaultKinematics, posOut, nrmOut, vertBase);
    }

}// namespace threepp::fauna

#endif// THREEPP_EXTRAS_FAUNA_BIRDGEOMETRY_HPP
