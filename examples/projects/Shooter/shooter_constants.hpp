// ============================================================================
//  Shooter — tuning constants, shared RNG, and simple math helpers
//  Included inside namespace {} in main.cpp — not a standalone header.
// ============================================================================

// ---- tuning constants --------------------------------------------------
constexpr float kArena = 28.f;// half-extent of the play area
constexpr float kPlayerRadius = 0.35f;
constexpr float kPlayerLen = 1.1f;                              // capsule cylinder segment
constexpr float kPlayerHalf = kPlayerLen * 0.5f + kPlayerRadius;// centre->foot
constexpr float kWalkSpeed = 2.1f;
constexpr float kRunSpeed = 6.4f;
constexpr float kJumpSpeed = 5.2f;
constexpr float kMouseSens = 0.0026f;
constexpr float kFireInterval = 0.11f;
constexpr float kReloadTime = 1.25f;
constexpr int kMagSize = 12;
constexpr int kMaxDecals = 48;// bullet-impact decals before the oldest recycles
constexpr float kEnemySpeed = 2.3f;
constexpr int kEnemyHp = 3;
constexpr int kMaxEnemies = 6;
constexpr float kEnemyAttackRange = 1.7f;

// ---- grenade -----------------------------------------------------------
constexpr float kThrowTime = 1.1f;   // throw animation + cooldown
constexpr float kThrowRelease = 0.75f;// seconds into the throw when it leaves the hand
constexpr float kGrenadeSpeed = 13.f;// launch speed (m/s) along aim
constexpr float kGrenadeFuse = 1.4f; // seconds to detonation
constexpr float kBlastRadius = 4.5f; // explosion kill/knockback radius

// ---- camera (over-the-shoulder third person + right-click ADS zoom) ----
constexpr float kCamShoulder = 0.7f;    // hip over-the-shoulder offset (screen-right, m)
constexpr float kCamShoulderAds = 0.5f; // tighter shoulder while aiming
constexpr float kCamDistAds = 2.6f;     // camera pull-in distance while aiming
constexpr float kFovHip = 70.f;         // base vertical FOV (matches the camera ctor)
constexpr float kFovAds = 50.f;         // zoomed FOV while aiming
constexpr float kZoomSpeed = 12.f;      // ADS ease-in/out rate (per second)
constexpr float kInspectSpeed = 8.f;    // middle-mouse face-the-player swing rate
// Camera-wall collision: keep the boom from clipping through level geometry.
constexpr float kCamMinDist = 0.6f;     // closest the camera may pull toward the player
constexpr float kCamSkin = 0.25f;       // stop short of the wall so the near plane clears it
constexpr float kCamReturnSpeed = 6.f;  // ease-out rate once the obstruction passes
// Upper-body aim tilt: the spine is pitched by (aim pitch × gain) so the held
// rifle tracks the target vertically while both hands stay on it. 1.0 = gun
// matches the aim; lower = less lean. Flip the sign if the torso bends the
// wrong way (depends on the imported bone axes).
constexpr float kSpinePitchGain = 1.0f;

// ---- recoil ------------------------------------------------------------
constexpr float kRecoilPerShot = 0.015f;// rad of upward aim kick per shot
constexpr float kRecoilMax = 0.13f;     // cap on accumulated kick (~7.5 deg)
constexpr float kRecoilYawKick = 0.009f;// rad of random horizontal kick per shot
constexpr float kRecoilRecover = 9.f;   // recovery rate toward zero (per second)

// ---- enemy navigation (flow-field grid; built after the props are placed)
constexpr float kNavCell = 1.0f;    // grid cell size (m)
constexpr float kSeparation = 1.4f; // bots ease apart within this distance (m)

// ---- death ragdoll -----------------------------------------------------
constexpr float kRagdollTtl = 30.0f;// seconds a corpse ragdoll lingers before removal

// ---- SWAT player tuning (assets/swat.glb, built by scripts/mixamo_to_glb.py)
// The player is the Mixamo "Ch15" SWAT model with a full rifle-handling clip
// set (aim / fire / reload / strafe / run / jump / hit). The model faces +Z,
// so we spin the rig to camera-forward; flip by ±PI if it ends up back-to-front.
constexpr float kModelYaw = 0.f;
constexpr float kCharHeight = 1.7f;// target skeleton span (≈ standing height, metres)

// ---- palette -----------------------------------------------------------
constexpr int kHudCyan = 0x35c2ff;
constexpr int kHudGood = 0x47e07a;
constexpr int kHudWarn = 0xff4d4d;
constexpr int kPanel = 0x0e1b2a;
constexpr int kPanelEdge = 0x1d3b57;

// ---- HUD scale -----------------------------------------------------------
// Every HUD dimension here is a design unit: a logical pixel at 100% (96
// dpi). GLFW window coordinates are physical pixels on Windows/X11, so the
// overlay must scale by the monitor content scale or it draws half-size on
// a 200% display while text (formerly the only thing scaled) stays large —
// hence overlapping widgets. The whole overlay lives in two coordinate
// systems: SVG meshes render through `uiCam` (group transforms apply), so a
// widget GROUP is scaled by uiScale and its baked geometry + child offsets
// ride along; screen-space sprites (TextSprite, hit targets) bypass the
// scene graph — the renderer composes their matrix from screenAnchor +
// position — so makeText and the explicit sprite scales below carry uiScale
// themselves. On macOS window coords are already logical points (the
// renderer compensates via pixelRatio), so this stays 1.
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
