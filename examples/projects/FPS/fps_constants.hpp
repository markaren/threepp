// ============================================================================
//  FPS demo — tuning constants, shared RNG, and simple math helpers
//  Included inside namespace {} in main.cpp — not a standalone header.
// ============================================================================

// ---- arena ---------------------------------------------------------------
constexpr float kArena = 26.f;// half-extent of the play area

// ---- player --------------------------------------------------------------
constexpr float kPlayerRadius = 0.35f;
constexpr float kPlayerLen = 1.1f;                              // capsule cylinder segment
constexpr float kPlayerHalf = kPlayerLen * 0.5f + kPlayerRadius;// centre->foot
constexpr float kEyeOffset = 0.62f;                             // eye above capsule centre
constexpr float kWalkSpeed = 4.2f;
constexpr float kRunSpeed = 7.2f;
// Real-world gravity (9.81) + this jump speed gave a ~1.5 m, 1.1 s-hang-time
// arc — floaty, "moon gravity" jumping. Arcade FPS gravity (~2x earth) plus a
// faster launch keeps a similar ~1.2 m apex (still clears the parkour step
// blocks near the corner towers) but cuts hang time to ~0.7 s.
constexpr float kGravity = 20.f;
constexpr float kJumpSpeed = 8.2f;
constexpr float kMouseSens = 0.0024f;

// ---- weapon (AK-12) --------------------------------------------------------
constexpr float kFireInterval = 0.105f;// ~570 rpm full auto
constexpr float kReloadTime = 2.4f;    // Reload clip is squeezed into this
constexpr float kReloadHold = 0.35f;   // ...minus this, held on the last frame
                                       // while the overlay weight fades out
// Fallback equip window, used only if the Equip clip is missing: normally the
// clip's OWN length is used (see vmEquipLen in main.cpp). The clip ends exactly
// on Idle's first pose — measured worst-channel delta 0.04 rad — so letting it
// run out and crossfading there is seamless. Cutting it short is not: at the
// old 2.1 s cut of the 2.42 s clip the pose was still 0.67 rad away from Idle,
// and the crossfade across that gap was a visible lurch on every spawn.
constexpr float kEquipTime = 2.1f;
constexpr float kEquipFade = 0.22f;// crossfade to Idle over this, after the clip ends
constexpr int kMagSize = 30;
constexpr int kMaxDecals = 64;// bullet-impact decals before the oldest recycles

// ---- recoil ----------------------------------------------------------------
constexpr float kRecoilPerShot = 0.011f;// rad of upward camera kick per shot
constexpr float kRecoilMax = 0.11f;     // cap on accumulated kick
constexpr float kRecoilYawKick = 0.006f;// rad of random horizontal kick per shot
constexpr float kRecoilRecover = 8.f;   // recovery rate toward zero (per second)

// ---- shooting range ---------------------------------------------------------
// The demo is a range, not a deathmatch: nothing shoots back. What it shows
// instead is IMPACT — every target is a real PhysX dynamic body with a mesh
// collider, so a hit transfers a genuine impulse at the contact point and the
// thing topples, spins or rolls the way its mass and shape say it should.
constexpr float kLaneCount = 3;
constexpr float kLaneWidth = 7.f;      // centre-to-centre spacing
constexpr float kFiringLineZ = -18.f;  // where the player starts, behind the bench
constexpr float kRangeNearZ = -6.f;    // nearest target row
constexpr float kRangeFarZ = 20.f;     // backstop
// Impulse per hit, scaled by the round's momentum rather than tuned per prop:
// a light plate flips, a loaded crate stack barely shifts, and neither needed
// a number typed for it.
constexpr float kBulletImpulse = 14.f;
// A plate counts as DOWN once it has tipped this far off vertical. Measured
// off the body's own up axis, so a plate knocked spinning still scores only
// when it actually falls.
constexpr float kPlateDownCos = 0.55f;// ~57 degrees
// Knocked-down plates pop back up after this, so the range keeps giving you
// something to shoot without a manual reset.
constexpr float kPlateResetDelay = 3.5f;
// Props that get shot off the range are recycled home rather than falling
// forever.
constexpr float kPropRecycleY = -8.f;

// ---- palette (HUD) ---------------------------------------------------------
constexpr int kHudCyan = 0x35c2ff;
constexpr int kHudGood = 0x47e07a;
constexpr int kHudWarn = 0xff4d4d;
constexpr int kPanel = 0x0e1b2a;
constexpr int kPanelEdge = 0x1d3b57;

// ---- HUD scale (see tps_shooter: physical-pixel windows need content scale) -
float uiScale = 1.f;

std::mt19937 rng{1337};
float frand(float a, float b) {
    return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(rng);
}

float wrapPi(float a) {
    while (a > math::PI) a -= 2.f * math::PI;
    while (a < -math::PI) a += 2.f * math::PI;
    return a;
}
