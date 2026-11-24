// Where everything inside the mill stands, and the gear train's numbers.
//
// The waterwheel's iron shaft comes through the west wall at the axle's height (3.25 m), runs in a bearing on a stone
// pier against the wall and carries, overhung past the bearing, the pit wheel: a cast iron mortise wheel with 48
// hornbeam cogs set in its east face. The axle is too close under the stone floor for the usual layout, so the cogs
// drive the wallower at the BOTTOM of the pit wheel: a lantern pinion of 14 iron staves on the upright shaft, which
// stands just past the end of the wheel's shaft (the two axes cross at right angles, as a face wheel and a lantern
// need). The upright, an octagonal oak shaft, rises from a footstep on the basement floor through all three floors.
// Over the stone floor it carries the great spur wheel (84 cogs, clasp arms) driving two cast iron stone nuts of 16
// teeth on the spindles of two pairs of stones below them ("overdrift": the runners hang from their spindles, whose
// top bearings are in a bridge beam under the bin floor). At its head in the garner the crown wheel (40 cogs facing
// up) drives a lantern of 10 staves on the lay shaft, whose belt pulley drives the sack hoist through a slack belt
// (the hoist is out of gear: the belt hangs loose).
//
// Speeds at the wheel's 5.5 rpm: upright 5.5 * 48/14 = 18.9 rpm; runners 18.9 * 84/16 = 99 rpm; lay shaft 75 rpm.

#ifndef THREEPP_EXAMPLE_WATERMILL_INTERIORLAYOUT_HPP
#define THREEPP_EXAMPLE_WATERMILL_INTERIORLAYOUT_HPP

#include "Site.hpp"

#include <array>

namespace mill::inside {

    constexpr float pi = 3.14159265f;
    constexpr float tau = 6.2831853f;

    // ---- tooth counts ---------------------------------------------------------------------------------------------
    constexpr int pitCogs = 48, wallowerStaves = 14, spurCogs = 84, nutTeeth = 16, crownCogs = 40, layStaves = 10;

    // ---- the pit wheel (on the waterwheel's shaft; axial positions are world x) ------------------------------------
    constexpr float axleY = site::wheelY, axleZ = site::wheelZ;
    constexpr float pitX = 7.2f;                   // the rim's middle plane
    constexpr float pitRimHalf = 0.07f;            // the rim is 0.14 thick along x
    constexpr float pitFace = pitX + pitRimHalf;   // the cogs stand out of this face (east)
    constexpr float pitCogOut = 0.07f;             // how far (short, as lantern gearing wants: a stave's width and a half)
    constexpr float pitR = 0.96f;                  // the cogs' mean radius
    constexpr float pitCogHalf = 0.06f;            // their radial half length (0.90 .. 1.02)
    constexpr float pitRim0 = 0.80f, pitRim1 = 1.08f;
    constexpr float shaftEndX = 7.37f;             // the wheel's shaft ends here, short of the upright
    constexpr float bearingX = 6.8f;               // its inner bearing, on a pier against the wall

    // ---- the wallower and the upright shaft ----------------------------------------------------------------------------
    constexpr float wallowerR = pitR * wallowerStaves / pitCogs;      // 0.28: equal pitch with the pit wheel's cogs
    constexpr float uprightX = pitFace + 0.5f * pitCogOut + wallowerR;// 7.615: the staves run through the cogs' middle
    constexpr float uprightZ = axleZ;
    constexpr float uprightR = 0.17f;                                 // to the octagon's flats
    constexpr float uprightFoot = 1.05f, uprightHead = 10.1f;
    // the lantern's staves span the cogs' radial length at the pit wheel's bottom (y 2.23 .. 2.35)
    constexpr float wallowerY0 = 2.18f, wallowerY1 = 2.40f, wallowerHead = 0.07f, wallowerHeadR = 0.33f, wallowerStave = 0.022f;

    // ---- the great spur wheel and the stone nuts -------------------------------------------------------------------------
    constexpr float spurPitch = 0.09f;
    constexpr float spurR = spurCogs * spurPitch / tau;// 1.203, the pitch circle
    constexpr float nutR = nutTeeth * spurPitch / tau; // 0.229
    constexpr float spindleD = spurR + nutR;           // 1.432, upright to spindle
    constexpr float spurY0 = 6.18f, spurY1 = 6.34f;
    constexpr float spurRim0 = 0.98f, spurRim1 = spurR - 0.036f;// cogs from the rim out to spurR + 0.032
    // the stones lie south-east and north-east of the upright (angle psi about +y from +x: dir = (cos, 0, -sin))
    constexpr std::array<float, 2> stoneAngle{-pi / 3.f, pi / 3.f};

    inline float spindleX(int i) { return uprightX + spindleD * std::cos(stoneAngle[i]); }
    inline float spindleZ(int i) { return uprightZ - spindleD * std::sin(stoneAngle[i]); }

    // ---- the stones --------------------------------------------------------------------------------------------------
    constexpr float stoneRadius = 0.6f, eyeRadius = 0.13f;
    constexpr float hurstTop = 4.8f;                    // the low timber platform the bed stones lie on
    constexpr float bedTop = 5.08f;                     // the bed stone's dressed face
    constexpr float runner0 = 5.086f, runner1 = 5.366f; // the runner, face down
    constexpr float tunR = 0.72f, tunTop = 5.45f, lidTop = 5.48f;
    constexpr float hopperOut = 0.4f;                   // the hopper stands this far out from the spindle (away from the upright)
    constexpr float damsel0 = 5.52f, damsel1 = 5.66f;
    constexpr float nutY0 = spurY0, nutY1 = spurY1;

    // ---- the crown wheel and the lay shaft ------------------------------------------------------------------------------
    constexpr float crownR = crownCogs * spurPitch / tau;// 0.573
    constexpr float layR = layStaves * spurPitch / tau;  // 0.143
    constexpr float crownFace = 9.87f, crownCogOut = 0.06f, crownCogHalf = 0.05f;
    constexpr float layY = crownFace + 0.5f * crownCogOut + layR;// 10.073
    constexpr float layLantern0 = uprightX + crownR - 0.09f, layLantern1 = uprightX + crownR + 0.09f;
    constexpr float layX0 = layLantern0 - 0.04f, layX1 = 10.45f;
    constexpr float layHeadR = layR + 0.03f, layStave = 0.015f;
    constexpr float pulleyX = 10.25f, layPulleyR = 0.28f;
    constexpr float hoistZ = -1.4f, hoistY = 10.68f, hoistPulleyR = 0.45f, drumR = 0.15f;

    // ---- floors -------------------------------------------------------------------------------------------------------
    constexpr float boardT = 0.035f, joistD = 0.2f, joistW = 0.09f, beamD = 0.3f, beamW = 0.28f;
    constexpr float beamZ = 1.95f, midPostX = 9.6f, postW = 0.2f;

    struct Rect {
        float x0, x1, z0, z1;
        [[nodiscard]] bool overlapsX(float a, float b) const { return b > x0 && a < x1; }
        [[nodiscard]] bool overlapsZ(float a, float b) const { return b > z0 && a < z1; }
    };

    // holes through the floors
    inline Rect uprightHole() { return {uprightX - 0.25f, uprightX + 0.25f, uprightZ - 0.25f, uprightZ + 0.25f}; }
    inline Rect sackHatch() { return {10.9f, 11.6f, -1.7f, -1.1f}; }
    // the basement stair (along the north wall, rising west) comes up through the stone floor here
    inline Rect basementWell() { return {9.3f, 10.8f, -3.9f, -3.1f}; }
    // stair 1 (stone floor to bin floor, along the south wall, rising east)
    inline Rect binWell() { return {9.2f, 11.07f, 3.42f, 4.38f}; }
    // stair 2 (bin floor to garner, along the north wall, rising west)
    inline Rect garnerWell() { return {8.2f, 10.05f, -4.38f, -3.42f}; }

}// namespace mill::inside

#endif
