// The watermill's site: where everything stands, and the ground's height anywhere.
//
// Metres, y up. The stream runs toward +z (downstream). The mill stands on the right bank (+x) with its long west wall
// along the stream and its south gable (+z) facing downstream; the overshot wheel turns in the plane x = wheelX
// between the mill's wall and a stone pier, fed at its top by a flume on a timber trestle that comes down the right
// bank from upstream (-z). A footbridge crosses the stream upstream of the wheel on two stone piers, passing under the
// flume at its right-hand end. The hero view stands on the rocks downstream, low, looking upstream, as the reference
// photograph does.

#ifndef THREEPP_EXAMPLE_WATERMILL_SITE_HPP
#define THREEPP_EXAMPLE_WATERMILL_SITE_HPP

#include "Noise.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace mill::site {

    // ---- the mill ---------------------------------------------------------------------------------------------------
    constexpr float millWest = 6.0f, millEast = 12.5f;  // x of the outer faces of the timber walls
    constexpr float millNorth = -4.5f, millSouth = 4.5f;// z of the gable walls' outer faces
    constexpr float foundationThickness = 0.6f;          // the stone walls of the basement
    constexpr float basementFloor = 0.9f;                // the spur floor (packed earth and flags)
    constexpr float sillTop = 4.3f;                      // top of the stone foundation, under the timber sill
    constexpr float stoneFloor = 4.4f;                   // the first timber floor: the millstones stand here
    constexpr float binFloor = 7.0f;                     // the second floor: bins and the bolter
    constexpr float garnerFloor = 9.6f;                  // the third floor, under the roof: the crown wheel, the sack hoist
    constexpr float eaves = 11.2f;                       // top of the wall plates
    constexpr float roofPitch = 0.785398f;               // 45 degrees
    constexpr float ridge = eaves + 0.5f * (millEast - millWest) * 1.f;
    constexpr float wallThickness = 0.18f;               // studs and boards of the timber walls

    // ---- the wheel ---------------------------------------------------------------------------------------------------
    constexpr float wheelX = 5.15f;    // the wheel's middle plane
    constexpr float wheelY = 3.25f;    // its axle
    constexpr float wheelZ = 0.5f;
    constexpr float wheelRadius = 2.9f;// to the rim's outside
    constexpr float wheelWidth = 1.0f; // between the shrouds' outsides
    constexpr int wheelBuckets = 36;
    // the stone pier carrying the axle's outer bearing
    constexpr float pierX0 = 3.75f, pierX1 = 4.45f, pierZ0 = wheelZ - 0.9f, pierZ1 = wheelZ + 0.9f;
    // the axle runs from the pier's bearing through the wall to the pit wheel inside
    constexpr float axleOuterX = 4.1f, axleInnerX = 7.3f;

    // ---- the flume ---------------------------------------------------------------------------------------------------
    constexpr float flumeX = wheelX;
    constexpr float flumeInner = 0.8f;                         // between its walls
    constexpr float flumeWall = 0.45f;                         // height of its walls over its floor
    constexpr float flumeEndZ = wheelZ + 0.45f;                // its mouth, a little past the wheel's top
    constexpr float flumeFloorAtEnd = wheelY + wheelRadius + 0.28f;
    constexpr float flumeSlope = 0.004f;                       // rising upstream
    // As at Glade Creek, the flume is short and starts in a head box: a timber box on a tall stone base just below the
    // footbridge's landing, which the water fills from a pipe buried in the bank (fed from the creek higher up), so
    // it seems to come from nowhere. The trough leaves the box's downstream side.
    constexpr float headZ0 = -11.75f, headZ1 = -9.95f, headHalfX = 0.85f;
    constexpr float flumeStartZ = headZ0 + 0.2f;               // the trough starts inside the head box
    // the trough has settled on its trestle: it undulates along its length (always still falling to the wheel), true
    // only over its last few metres, where the wheel's bracket holds it
    inline float flumeFloor(float z) {
        const float wave = 0.035f * fbm2(z * 0.06f, 2.7f, 3, 61u) * smooth(0.f, 5.f, flumeEndZ - z);
        return flumeFloorAtEnd + flumeSlope * (flumeEndZ - z) + wave;
    }

    // ---- the bridge --------------------------------------------------------------------------------------------------
    constexpr float bridgeZ = -13.f;
    constexpr float deckTop = 3.9f;
    constexpr float bridgeX0 = -17.f, bridgeX1 = 7.6f;
    constexpr float deckWidth = 1.7f;
    constexpr std::array<float, 2> bridgePiers{-8.5f, -1.0f};
    constexpr float pierHalfX = 0.8f, pierHalfZ = 1.15f;

    // ---- the stream --------------------------------------------------------------------------------------------------
    // its centre line and half width along z (Catmull-Rom through control points)
    struct Knot {
        float z, cx, hw;
    };
    // The creek winds: upstream it comes round a bend from the left (so from the mill nobody sees where it starts),
    // swings right past the mill and the falls, and bends away left again below the pool.
    inline const std::array<Knot, 15>& streamKnots() {
        static const std::array<Knot, 15> k{{{-120.f, -22.f, 4.5f},
                                             {-90.f, -17.f, 5.f},
                                             {-66.f, -13.5f, 5.2f},
                                             {-50.f, -10.f, 5.4f},
                                             {-38.f, -3.8f, 5.9f},
                                             {-26.f, -0.4f, 6.3f},
                                             {-13.f, -4.4f, 6.6f},
                                             {-3.f, -2.6f, 6.0f},
                                             {8.f, -1.3f, 6.5f},
                                             {17.f, 0.5f, 8.0f},
                                             {27.f, 0.9f, 8.3f},
                                             {37.f, 0.9f, 8.0f},
                                             {47.f, -1.4f, 8.6f},
                                             {62.f, -7.0f, 7.5f},
                                             {95.f, -16.f, 7.0f}}};
        return k;
    }

    inline float catmull(float p0, float p1, float p2, float p3, float t) {
        const float t2 = t * t, t3 = t2 * t;
        return 0.5f * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
    }

    inline void streamAt(float z, float& cx, float& hw) {
        const auto& k = streamKnots();
        const int n = static_cast<int>(k.size());
        if (z <= k[0].z) {
            cx = k[0].cx;
            hw = k[0].hw;
            return;
        }
        if (z >= k[n - 1].z) {
            cx = k[n - 1].cx;
            hw = k[n - 1].hw;
            return;
        }
        int i = 0;
        while (i + 1 < n && k[i + 1].z < z) ++i;
        const int i0 = std::max(i - 1, 0), i2 = std::min(i + 1, n - 1), i3 = std::min(i + 2, n - 1);
        const float t = (z - k[i].z) / (k[i2].z - k[i].z);
        cx = catmull(k[i0].cx, k[i].cx, k[i2].cx, k[i3].cx, t);
        hw = catmull(k[i0].hw, k[i].hw, k[i2].hw, k[i3].hw, t);
    }

    // a step down of `drop` m at z0 (the stream's ledges)
    inline float ledge(float z, float z0, float drop, float width = 0.5f) { return drop * (1.f - smooth(z0 - width, z0 + width, z)); }

    constexpr float ledgeUpper = -20.f, ledgeLower = 7.f;
    // the falls below the mill: the creek goes over a sandstone ledge into a pool, as at Glade Creek
    constexpr float fallsZ = 34.5f, fallsDrop = 1.9f;

    // The ledges are beds of sandstone the creek has broken back unevenly: their edges run crooked and askew across
    // the channel. Where each one's edge is at a given x:
    inline float upperLedgeZ(float x) { return ledgeUpper + 1.7f * fbm2(x * 0.2f, 7.7f, 3, 41u) + 0.14f * (x + 2.f); }
    inline float lowerLedgeZ(float x) { return ledgeLower + 1.3f * fbm2(x * 0.24f, 9.1f, 3, 42u) - 0.11f * (x + 1.f); }
    // the falls' lip, cut back into a bay where the main chute goes over, ragged elsewhere
    inline float fallsLipZ(float x) { return fallsZ - 0.75f * std::exp(-((x - 1.4f) / 2.6f) * ((x - 1.4f) / 2.6f)) + 0.5f * fbm2(x * 0.32f, 11.3f, 3, 43u); }

    // The creek comes down in steps: a run of small cascades over sandstone beds and jammed boulders, a pool or a
    // riffle between each. Each step's edge runs crooked and askew across the channel. Steps above the mill raise the
    // bed upstream of them, steps below it lower the bed downstream, so the water at the wheel stays where it is.
    struct CreekStep {
        float z, drop, skew;
        uint32_t seed;
    };
    inline const std::array<CreekStep, 11>& creekSteps() {
        static const std::array<CreekStep, 11> s{{{-4.8f, 0.22f, 0.10f, 71u},
                                                  {-10.5f, 0.28f, -0.13f, 72u},
                                                  {-16.f, 0.18f, 0.06f, 73u},
                                                  {-26.5f, 0.32f, -0.09f, 74u},
                                                  {-32.f, 0.24f, 0.15f, 75u},
                                                  {-37.5f, 0.3f, -0.05f, 76u},
                                                  {-44.f, 0.34f, 0.11f, 77u},
                                                  {-52.f, 0.3f, -0.12f, 78u},
                                                  {-61.f, 0.35f, 0.08f, 79u},
                                                  {21.5f, 0.3f, -0.12f, 80u},
                                                  {27.2f, 0.22f, 0.14f, 81u}}};
        return s;
    }
    inline float stepEdgeZ(const CreekStep& s, float x) { return s.z + s.skew * (x + 1.f) + 1.1f * fbm2(x * 0.26f, 3.7f, 3, s.seed); }

    // the bed's middle at a point (no cross-section: the slope, the steps, the ledges and the falls)
    inline float bedAt(float x, float z) {
        // steeper above the mill, where the creek comes down through boulders
        float b = -0.65f - (z < 0.f ? 0.032f : 0.02f) * z;
        for (const auto& s : creekSteps()) {
            const float e = stepEdgeZ(s, x);
            if (s.z < 0.f) b += s.drop * (1.f - smooth(e - 0.3f, e + 0.3f, z));
            else b -= s.drop * smooth(e - 0.3f, e + 0.3f, z);
        }
        return b + ledge(z, upperLedgeZ(x), 0.35f) + ledge(z, lowerLedgeZ(x), 0.3f) - fallsDrop * smooth(fallsLipZ(x) - 0.1f, fallsLipZ(x) + 0.1f, z);
    }

    // the same along the creek's middle line
    inline float bedProfile(float z) {
        float cx, hw;
        streamAt(z, cx, hw);
        return bedAt(cx, z);
    }

    // each bank wanders on its own: the half width on one side (side -1 left, +1 right) at z
    inline float bankHalfWidth(float z, float hw, float side) {
        const uint32_t s = side < 0 ? 51u : 52u;
        return hw * (1.f + 0.14f * fbm2(z * 0.07f, side * 3.3f, 3, s)) + 0.7f * fbm2(z * 0.28f, side * 5.1f, 2, s + 10u);
    }

    // the pool below the lower ledge, near the hero view
    inline float poolDepth(float x, float z) {
        const float dx = (x - 2.5f) / 6.5f, dz = (z - 16.5f) / 7.5f;
        // and the plunge pool under the falls, deeper and calm
        const float ex = (x - 1.f) / 7.5f, ez = (z - 41.5f) / 6.5f;
        return 0.6f * smooth(1.f, 0.3f, std::sqrt(dx * dx + dz * dz)) + 1.7f * smooth(1.f, 0.2f, std::sqrt(ex * ex + ez * ez));
    }

    // where the water stands, roughly (the simulation finds the real level)
    inline float nominalWater(float z) { return bedProfile(z) + 0.3f; }

    // ---- the ground --------------------------------------------------------------------------------------------------
    inline float mix1(float a, float b, float t) { return a + (b - a) * t; }

    inline bool inMill(float x, float z, float margin = 0.f) {
        return x > millWest - margin && x < millEast + margin && z > millNorth - margin && z < millSouth + margin;
    }

    // the wheel's pit: from the pier to the wall, the wheel's length and a tail downstream
    inline bool inWheelPit(float x, float z) {
        return x > pierX0 - 0.3f && x < millWest + 0.05f && z > wheelZ - wheelRadius - 0.4f && z < wheelZ + wheelRadius + 1.2f;
    }

    // the ground, the stream's bed included (no rocks: they are their own meshes)
    inline float groundHeight(float x, float z) {
        float cx, hw;
        streamAt(z, cx, hw);
        const float base = bedAt(x, z);
        const float d = x - cx;
        const float hs = bankHalfWidth(z, hw, d < 0 ? -1.f : 1.f);
        const float e = std::abs(d) - hs;// > 0 outside the channel
        const float rough = 0.06f * fbm2(x * 0.7f, z * 0.7f, 3, 11u);
        float h;
        if (e <= 0) {
            // the deepest line wanders from side to side; gravel bars build where it has left; the bed is ribbed
            // across the flow by riffles and scoured into hollows
            const float t = d / hs;
            const float thalweg = 0.45f * fbm2(z * 0.035f, 3.1f, 2, 53u);
            const float bar = 0.16f * smooth(0.2f, 0.8f, std::abs(t - thalweg)) * smooth(-0.1f, 0.4f, fbm2(z * 0.09f, t * 0.5f, 2, 54u));
            const float riffle = 0.06f * fbm2(x * 0.35f, z * 0.9f, 3, 55u) + 0.07f * fbm2(x * 0.18f, z * 0.18f, 3, 56u);
            h = base - 0.04f + 0.3f * (t - thalweg) * (t - thalweg) + bar + riffle + rough;
        } else if (d < 0) {
            // the left bank: up into the woods, then the valley side
            const float lumps = 0.5f * fbm2(x * 0.12f, z * 0.12f, 4, 12u) * smooth(1.f, 6.f, e);
            h = base + 0.22f + 1.1f * smooth(0.f, 3.f, e) + 0.12f * e + 0.0045f * e * e + lumps;
        } else {
            // the right bank: the mill's terrace, the hill behind it, the rocks downstream
            const float lumps = 0.5f * fbm2(x * 0.12f, z * 0.12f, 4, 13u) * smooth(1.f, 6.f, e);
            const float bank = 0.9f * smooth(0.f, 2.f, e) + 0.22f * e + 0.006f * e * e;
            // the terrace the mill is cut into: level ground at 2.6 m east of the wheel, from z -9 to 9
            const float terraceW = smooth(-14.f, -8.f, z) * (1.f - smooth(4.6f, 8.f, z)) * smooth(7.5f, 10.5f, x);
            const float terrace = 2.6f + 0.02f * std::max(0.f, x - 14.f) * std::max(0.f, x - 14.f) + 0.4f * std::max(0.f, x - 14.f);
            h = base + 0.22f + mix1(bank, terrace - base - 0.22f, terraceW * smooth(3.f, 6.5f, x)) + lumps;
            // downstream: the bank behind the sandstone outcrop (the outcrop's slabs are rocks) stands high over the pool
            const float cliff = smooth(6.f, 10.f, z) * smooth(2.5f, 5.f, e);
            h += cliff * (2.6f + 0.2f * std::min(e, 12.f));
        }
        // the mill's footprint is its basement: the ground stays under the basement floor there
        if (inMill(x, z, -0.3f)) h = std::min(h, basementFloor - 0.1f);
        // the yard in front of the basement door: level just under its sill, rising to the bank round it
        {
            const float dx = std::max({millWest - 0.5f - x, 0.f, x - (millWest + 3.2f)}), dz = std::max(0.f, z - (millSouth + 2.2f));
            if (z > millSouth - 0.5f) h = std::min(h, basementFloor - 0.08f + 0.7f * std::sqrt(dx * dx + dz * dz) + 0.06f * fbm2(x * 1.3f, z * 1.3f, 2, 17u));
        }
        // the wheel's pit: its floor under the wheel, its edges sloping up to the ground round it
        {
            const float px0 = pierX0 - 0.2f, px1 = millWest + 0.2f, pz0 = wheelZ - wheelRadius - 0.3f, pz1 = wheelZ + wheelRadius + 1.0f;
            const float dx = std::max({px0 - x, 0.f, x - px1}), dz = std::max({pz0 - z, 0.f, z - pz1});
            h = std::min(h, base - 0.05f + 1.1f * std::sqrt(dx * dx + dz * dz));
        }
        h -= poolDepth(x, z);
        return h;
    }

    // the hero view (the reference photograph)
    struct View {
        float ex, ey, ez, tx, ty, tz, fov;
    };

}// namespace mill::site

#endif
