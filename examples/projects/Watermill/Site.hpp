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
    constexpr float flumeStartZ = -46.f;                       // it comes out of the woods here
    inline float flumeFloor(float z) { return flumeFloorAtEnd + flumeSlope * (flumeEndZ - z); }

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
    inline const std::array<Knot, 13>& streamKnots() {
        static const std::array<Knot, 13> k{{{-110.f, -9.f, 4.5f},
                                             {-70.f, -6.f, 5.f},
                                             {-45.f, -4.5f, 5.5f},
                                             {-30.f, -2.6f, 6.0f},
                                             {-13.f, -2.4f, 6.6f},
                                             {0.f, -2.0f, 6.0f},
                                             {8.f, -1.2f, 6.6f},
                                             {16.f, 0.6f, 8.2f},
                                             {28.f, 0.8f, 8.4f},
                                             {36.f, 0.6f, 8.0f},
                                             {45.f, 0.4f, 9.0f},
                                             {60.f, -1.5f, 8.0f},
                                             {90.f, -4.0f, 7.0f}}};
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

    // the bed's middle along the stream
    inline float bedProfile(float z) {
        return -0.65f - 0.02f * z + ledge(z, ledgeUpper, 0.35f) + ledge(z, ledgeLower, 0.3f) - fallsDrop * smooth(fallsZ - 0.1f, fallsZ + 0.1f, z);
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
        const float base = bedProfile(z);
        const float d = x - cx;
        const float e = std::abs(d) - hw;// > 0 outside the channel
        const float rough = 0.06f * fbm2(x * 0.7f, z * 0.7f, 3, 11u);
        float h;
        if (e <= 0) {
            const float t = d / hw;
            h = base + 0.22f * t * t + rough;
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
