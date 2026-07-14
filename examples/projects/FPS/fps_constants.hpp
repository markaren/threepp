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
constexpr float kEquipTime = 2.1f;     // Equip clip plays this long on spawn
constexpr int kMagSize = 30;
constexpr int kMaxDecals = 64;// bullet-impact decals before the oldest recycles

// ---- recoil ----------------------------------------------------------------
constexpr float kRecoilPerShot = 0.011f;// rad of upward camera kick per shot
constexpr float kRecoilMax = 0.11f;     // cap on accumulated kick
constexpr float kRecoilYawKick = 0.006f;// rad of random horizontal kick per shot
constexpr float kRecoilRecover = 8.f;   // recovery rate toward zero (per second)

// ---- enemies ---------------------------------------------------------------
constexpr float kEnemySpeed = 3.0f;
constexpr int kEnemyHp = 4;
constexpr int kMaxEnemies = 4;
constexpr float kEnemyFireRange = 17.f; // stops and shoots inside this (with LOS)
constexpr float kEnemyFireInterval = 1.6f;
constexpr float kEnemyBurst = 3;        // shots per burst
constexpr float kEnemyBurstGap = 0.13f; // interval inside a burst
constexpr int kEnemyDamage = 4;
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
