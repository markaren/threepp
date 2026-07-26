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

// ---- enemies ---------------------------------------------------------------
constexpr float kEnemyRunSpeed = 3.6f; // closing the distance / breaking contact
constexpr float kEnemyWalkSpeed = 2.1f;// holding the fire band, strafing
constexpr int kEnemyHp = 4;
constexpr int kMaxEnemies = 4;
constexpr float kEnemyFireRange = 17.f;// engages inside this (with LOS)
constexpr float kEnemyFireInterval = 1.6f;
constexpr float kEnemyBurst = 3;        // shots per burst
constexpr float kEnemyBurstGap = 0.13f; // interval inside a burst
constexpr int kEnemyDamage = 4;
// ---- enemy behaviour -------------------------------------------------------
// The bots used to be a two-state machine: beeline at the player, then stop
// dead and plink from wherever they happened to arrive. These are the timers
// for the three-state version (Advance / Engage / Reposition) in main.cpp.
constexpr float kEnemyReaction = 0.42f; // s of unbroken LOS before the first burst
constexpr float kEnemyIdealMin = 6.5f;  // engagement band: back off inside this,
constexpr float kEnemyIdealMax = 12.5f; //  close in beyond this
constexpr float kEnemyStrafeMin = 0.8f; // s per strafe leg before flipping
constexpr float kEnemyStrafeMax = 2.2f;
constexpr float kEnemyRepositionOdds = 0.5f;// chance of breaking contact after a burst
constexpr float kEnemyRepositionTime = 1.8f;
constexpr float kEnemyPostRadiusMin = 4.5f;// where a reposition post is looked for
constexpr float kEnemyPostRadiusMax = 8.5f;
constexpr int kEnemyMag = 12;          // rounds before a (fire-blocking) reload
constexpr float kEnemyReloadTime = 2.2f;
// ---- authored stride speeds of the swat.glb locomotion clips (m/s) ---------
// Measured from each clip's Hips root-motion track: net horizontal hips
// displacement / clip duration, x the 0.01 Armature scale x the demo's
// kEnemyCharHeight rescale (0.979). Playing a clip at 1x while the bot travels
// at some OTHER speed is exactly what makes the feet slide — the walk clip runs
// at 0.97 m/s and the bots move at kEnemyWalkSpeed, a 2.2x mismatch, and
// "strafe right" is authored at half the speed of "strafe left". Every
// direction has a slow and a fast clip; the AI picks whichever sits closer to
// the speed actually being travelled and time-scales away the remainder.
constexpr float kClipWalk = 0.97f;      // "walking"            (+Y, forward)
constexpr float kClipRun = 3.08f;       // "rifle run"          (+Y)
constexpr float kClipWalkBack = 1.08f;  // "walking backwards"  (-Y)
constexpr float kClipRunBack = 2.52f;   // "run backwards"      (-Y)
constexpr float kClipStrafeL = 1.35f;   // "strafe left"        (+X, its left)
constexpr float kClipStrafeLFast = 2.75f;// "strafe (2)"        (+X)
constexpr float kClipStrafeR = 0.74f;   // "strafe right"       (-X)
constexpr float kClipStrafeRFast = 3.22f;// "strafe"            (-X)
constexpr float kGaitTimeScaleMin = 0.55f;// clamp on the residual time scale
constexpr float kGaitTimeScaleMax = 1.7f;
constexpr float kEnemyProbeDist = 1.9f;// local obstacle probe when steering off-field
constexpr float kRegenDelay = 4.f;     // seconds without damage before regen kicks in
constexpr float kRegenRate = 6.f;      // hp per second
constexpr float kEnemyCharHeight = 1.75f;// SWAT skeleton span (m)
// Unused pool rigs/rifles park here (kept visible: entry-list churn = deferred
// renderer structural rebuild — see the pool-creation note in main.cpp).
constexpr float kEnemyParkY = -80.f;
// Enemies are a single capsule collider (no per-bone hitboxes), so a headshot
// is approximated as any hit landing in the top slice of that capsule — the
// head+neck region on a 1.75 m SWAT frame.
constexpr float kHeadshotZone = 0.22f;

// ---- enemy navigation (flow-field grid; built after the props are placed)
constexpr float kNavCell = 1.0f;   // grid cell size (m)
constexpr float kSeparation = 1.6f;// bots ease apart within this distance (m)

// ---- death ragdoll ---------------------------------------------------------
constexpr float kRagdollTtl = 25.f;// seconds a corpse lingers before removal

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
