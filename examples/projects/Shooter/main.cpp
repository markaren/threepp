// ============================================================================
//  threepp Third-Person Shooter — a showcase prototype
// ============================================================================
//
//  Pulls together a lot of threepp at once:
//    * GLTF + skeletal animation (Soldier.glb) with idle/walk/run crossfades
//    * PhysX integration (extras/physx/PhysxWorld): a velocity-driven capsule
//      character, static level colliders, and dynamic crates/barrels you can
//      shoot and knock around. Shooting is a PhysX scene raycast.
//    * A pure-SVG heads-up display (crosshair, health, ammo, score, hit marker,
//      low-health vignette, game-over panel) drawn in an ortho overlay pass —
//      the same technique as examples/loaders/svg_ui.cpp.
//    * Procedurally synthesised placeholder sound effects (no asset files) via
//      the miniaudio-backed Audio API.
//
//  Controls:  WASD move   mouse look   LMB fire   RMB aim   MMB face player   SHIFT sprint   G grenade
//             SPACE jump   R reload     ENTER restart (when dead)   ESC quit
//
//  Everything here is a stand-in / prototype: the soldier is the placeholder
//  character, enemies are simple capsule bots, sounds are synthesised.
// ============================================================================

#include "threepp/threepp.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/audio/Audio.hpp"
#include "threepp/audio/WavFile.hpp"
#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/SpriteInteractor.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/DecalGeometry.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace ::physx;
namespace fs = std::filesystem;

// Minimal GLFW FFI for pointer-lock mouse-look. GLFW is compiled into the
// threepp library (as an OBJECT library), so these resolve at link time with
// no GLFW include / include-path needed. canvas.windowPtr() returns the
// GLFWwindow* (void*-compatible). Constants copied from GLFW/glfw3.h.
extern "C" {
void glfwSetInputMode(void* window, int mode, int value);
int glfwRawMouseMotionSupported(void);
}
namespace glfwc {
    constexpr int CURSOR = 0x00033001;
    constexpr int RAW_MOUSE_MOTION = 0x00033005;
    constexpr int CURSOR_NORMAL = 0x00034001;
    constexpr int CURSOR_DISABLED = 0x00034003;
    constexpr int TRUE_ = 1;
}// namespace glfwc

namespace {

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

    // ========================================================================
    //  Procedural placeholder sound effects
    // ========================================================================

    std::vector<float> synthShot(int sr = 44100) {
        const int n = sr * 18 / 100;// 0.18s
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float env = std::exp(-t * 22.f);
            const float noise = frand(-1.f, 1.f);
            const float thump = std::sin(2.f * math::PI * 70.f * t) * std::exp(-t * 12.f);
            s[i] = std::clamp((noise * 0.8f + thump * 0.6f) * env, -1.f, 1.f);
        }
        return s;
    }

    std::vector<float> synthClick(int sr = 44100) {
        const int n = sr * 3 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            s[i] = frand(-1.f, 1.f) * std::exp(-t * 120.f) * 0.5f;
        }
        return s;
    }

    std::vector<float> synthReload(int sr = 44100) {
        const int n = sr * 35 / 100;
        std::vector<float> s(n, 0.f);
        auto clickAt = [&](float at, float amp) {
            const int start = static_cast<int>(at * sr);
            for (int i = 0; i < sr * 3 / 100 && start + i < n; ++i) {
                const float t = static_cast<float>(i) / sr;
                s[start + i] += frand(-1.f, 1.f) * std::exp(-t * 90.f) * amp;
            }
        };
        clickAt(0.f, 0.5f);
        clickAt(0.16f, 0.4f);
        clickAt(0.30f, 0.6f);
        return s;
    }

    // ---- DSP helpers for the impact/step synths ----------------------------
    // One-pole low-pass; band-limited noise = lp(high cut) - lp(low cut).
    // Raw frand() noise reads as static hiss — every "physical" sound below
    // band-shapes it first.
    struct OnePole {
        float y = 0.f;
        float operator()(float x, float a) {
            y += a * (x - y);
            return y;
        }
    };
    float lpAlpha(float cutoffHz, int sr) {
        return 1.f - std::exp(-2.f * math::PI * cutoffHz / static_cast<float>(sr));
    }
    // Scale to a known peak so layering tweaks can't silently clip or vanish.
    std::vector<float> normalized(std::vector<float> s, float peak) {
        float m = 0.f;
        for (float x : s) m = std::max(m, std::abs(x));
        if (m > 1e-6f)
            for (float& x : s) x *= peak / m;
        return s;
    }

    // Bullet-into-body thwack: band-passed noise crack + a pitch-dropping body
    // thump + a duller low "wet" layer. Seeded so the bank renders a few
    // distinct variants (the old sound was a single 1.4 kHz sine ping — pure
    // arcade beep). Doubles as the hit-marker audio cue, which is why the
    // crack keeps some mid-band brightness.
    std::vector<float> synthHit(uint32_t seed, int sr = 44100) {
        std::mt19937 r(seed);
        auto rf = [&](float a, float b) { return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(r); };
        auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
        const int n = sr * 9 / 100;
        std::vector<float> s(n);
        const float f0 = rf(190.f, 240.f);
        OnePole lpHi, lpLo, lpWet;
        const float aHi = lpAlpha(rf(2600.f, 3400.f), sr);
        const float aLo = lpAlpha(800.f, sr);
        const float aWet = lpAlpha(550.f, sr);
        float phase = 0.f;
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float f = 80.f + f0 * std::exp(-t * 35.f);
            phase += 2.f * math::PI * f / sr;
            const float w = rn();
            const float thump = std::sin(phase) * std::exp(-t * 38.f) * 0.8f;
            const float crack = (lpHi(w, aHi) - lpLo(w, aLo)) * std::exp(-t * 70.f) * 1.6f;
            const float wet = lpWet(w, aWet) * std::exp(-t * 22.f) * 0.9f;
            s[i] = thump + crack + wet;
        }
        return normalized(std::move(s), 0.75f);
    }

    // Kill confirm: a deeper double knock (second hit ~70 ms behind the first)
    // with a low noise tail, so a lethal hit reads instantly different from a
    // normal one.
    std::vector<float> synthKill(uint32_t seed, int sr = 44100) {
        std::mt19937 r(seed);
        auto rf = [&](float a, float b) { return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(r); };
        auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
        const int n = sr * 30 / 100;
        std::vector<float> s(n, 0.f);
        auto knock = [&](float at, float f0, float amp) {
            const int start = static_cast<int>(at * sr);
            float phase = 0.f;
            for (int i = 0; start + i < n; ++i) {
                const float t = static_cast<float>(i) / sr;
                const float f = f0 * (0.45f + 0.55f * std::exp(-t * 28.f));
                phase += 2.f * math::PI * f / sr;
                s[start + i] += std::sin(phase) * std::exp(-t * 22.f) * amp;
            }
        };
        knock(0.f, rf(120.f, 145.f), 1.f);
        knock(rf(0.06f, 0.08f), rf(90.f, 110.f), 0.65f);
        OnePole lp;
        const float aLp = lpAlpha(420.f, sr);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            s[i] += lp(rn(), aLp) * std::exp(-t * 14.f) * 0.7f;
        }
        return normalized(std::move(s), 0.8f);
    }

    std::vector<float> synthThud(int sr = 44100) {
        const int n = sr * 14 / 100;
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float tone = std::sin(2.f * math::PI * 180.f * t);
            s[i] = (tone * 0.6f + frand(-1.f, 1.f) * 0.4f) * std::exp(-t * 24.f);
        }
        return s;
    }

    std::vector<float> synthBoom(int sr = 44100) {
        const int n = sr * 90 / 100;// 0.9s
        std::vector<float> s(n);
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float env = std::exp(-t * 5.f);
            const float sub = std::sin(2.f * math::PI * (90.f - 50.f * t) * t);// downward sweep
            const float crack = frand(-1.f, 1.f) * std::exp(-t * 18.f);        // initial crack
            const float rumble = frand(-1.f, 1.f) * env;
            s[i] = std::clamp((sub * 0.7f + crack * 0.6f + rumble * 0.4f) * env, -1.f, 1.f);
        }
        return s;
    }

    // Player-hit cue: a body thump under a falling tone (phase-accumulated so
    // the sweep is clean) with a slight vibrato and a breathy band-passed
    // layer — less "game-over beep" than the old bare descending sine.
    std::vector<float> synthHurt(int sr = 44100) {
        const int n = sr * 30 / 100;
        std::vector<float> s(n);
        OnePole lpHi, lpLo;
        const float aHi = lpAlpha(1400.f, sr);
        const float aLo = lpAlpha(500.f, sr);
        float phase = 0.f, phaseT = 0.f;
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            const float f = 150.f + 180.f * std::exp(-t * 7.f) + 12.f * std::sin(2.f * math::PI * 9.f * t);
            phaseT += 2.f * math::PI * f / sr;
            const float tone = std::sin(phaseT) * std::exp(-t * 8.f) * 0.7f;
            phase += 2.f * math::PI * (70.f + 100.f * std::exp(-t * 40.f)) / sr;
            const float thump = std::sin(phase) * std::exp(-t * 30.f) * 0.8f;
            const float w = frand(-1.f, 1.f);
            const float breath = (lpHi(w, aHi) - lpLo(w, aLo)) * std::exp(-t * 12.f) * 0.6f;
            s[i] = tone + thump + breath;
        }
        return normalized(std::move(s), 0.7f);
    }

    // One footstep on gritty sand/concrete: a soft pitch-dropping heel thump,
    // then a band-passed scuff whose amplitude is re-modulated by slow noise
    // (the "grit" crunch). Seeded — the bank renders several distinct variants
    // and cycles them, so successive steps never replay one identical sample
    // (the old step was a single 70 ms white-noise tick).
    std::vector<float> synthStep(uint32_t seed, int sr = 44100) {
        std::mt19937 r(seed);
        auto rf = [&](float a, float b) { return a + (b - a) * std::uniform_real_distribution<float>(0.f, 1.f)(r); };
        auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
        const int n = static_cast<int>(static_cast<float>(sr) * rf(0.10f, 0.13f));
        std::vector<float> s(n);
        const float f0 = rf(110.f, 150.f);
        const float scuffAt = rf(0.008f, 0.018f);// sole contact lags the heel strike
        OnePole lpHi, lpHi2, lpLo, grit;
        const float aHi = lpAlpha(rf(1900.f, 2700.f), sr);
        const float aLo = lpAlpha(rf(320.f, 460.f), sr);
        const float aGrit = lpAlpha(rf(60.f, 90.f), sr);
        float phase = 0.f;
        for (int i = 0; i < n; ++i) {
            const float t = static_cast<float>(i) / sr;
            phase += 2.f * math::PI * (55.f + f0 * std::exp(-t * 30.f)) / sr;
            const float heel = std::sin(phase) * std::exp(-t * 45.f) * 0.8f;
            const float w = rn();
            // cascade the high cut (12 dB/oct): one pole leaves enough leakage
            // above the cutoff that the scuff hisses instead of crunching
            const float band = lpHi2(lpHi(w, aHi), aHi) - lpLo(w, aLo);
            // slow-noise modulation makes the scuff crunch instead of hiss
            const float tex = std::clamp(0.35f + 9.f * std::abs(grit(rn(), aGrit)), 0.f, 1.f);
            const float ts = t - scuffAt;
            const float scuff = ts > 0.f ? band * tex * std::exp(-ts * 32.f) * 1.4f : 0.f;
            s[i] = heel + scuff;
        }
        return normalized(std::move(s), 0.6f);
    }

    // A pooled, retriggerable sound: round-robins a few voices so rapid fire
    // overlaps instead of cutting itself off. Voices may hold different synth
    // variants of the same sound, so the rotation also cycles variants. Each
    // play() takes a volume scale + playback rate — re-rolling those per
    // trigger is what keeps repeated one-shots (steps, hits) from sounding
    // machine-gunned. Degrades to a no-op if the audio device or file failed
    // to initialise.
    struct Sound {
        std::vector<std::unique_ptr<Audio>> voices;
        size_t next = 0;
        float volume = 0.6f;
        void play(float volScale = 1.f, float rate = 1.f) {
            if (voices.empty()) return;
            auto& v = voices[next];
            v->stop();
            v->seekToStart();// rewind so re-fire restarts from frame 0
            v->setVolume(volume * volScale);
            v->setPlaybackRate(rate);
            v->play();
            next = (next + 1) % voices.size();
        }
    };

    // Like Sound, but spatialised: playAt() drops the source at a world position so
    // the blast pans + attenuates relative to the camera-mounted AudioListener.
    struct PositionalSound {
        std::vector<std::unique_ptr<PositionalAudio>> voices;
        size_t next = 0;
        void playAt(const Vector3& p, float rate = 1.f) {
            if (voices.empty()) return;
            auto& v = voices[next];
            v->stop();
            v->seekToStart();
            v->setPlaybackRate(rate);
            v->position.copy(p);
            v->updateMatrixWorld(true);// push the new source position to the audio engine
            v->play();
            next = (next + 1) % voices.size();
        }
    };

    struct SoundBank {
        std::unique_ptr<AudioListener> listener;
        Sound shot, empty, reload, hit, thud, hurt, step, metal;
        PositionalSound boom;// grenade blast — spatialised at the detonation point
        bool ok = false;

        void init(Object3D& attachTo) {
            try {
                const fs::path dir = fs::temp_directory_path() / "threepp_tps_sounds";
                fs::create_directories(dir);
                struct Spec {
                    const char* name;                         // temp WAV base name (synth fallback)
                    std::vector<std::vector<float>> variants;// synth renders; voice i loads variant i % N
                    Sound* dst;
                    int voices;
                    std::string file;// external audio file; used instead of synth when set
                };

                // Real submachine-gun sample for the gun; the rest stay
                // procedural. Falls back to the synth shot if the file is absent.
                const std::string assets = std::string(DATA_FOLDER) + "/sounds/";
                const std::string gunFile = assets + "freesound_community-submachine-gun-79846.mp3";
                const std::string reloadFile = assets + "freesound_community-1911-reload-6248.mp3";
                const std::string metalFile = assets + "freesound_community-hard-metal-impact-43052.mp3";
                const std::string boomFile = assets + "grenade_explosion.mp3";
                std::vector<Spec> specs{
                        {"shot", {synthShot()}, &shot, 6, fs::exists(gunFile) ? gunFile : std::string{}},
                        {"empty", {synthClick()}, &empty, 2, {}},
                        {"reload", {synthReload()}, &reload, 2, fs::exists(reloadFile) ? reloadFile : std::string{}},
                        {"hit", {synthHit(11), synthHit(22), synthHit(33)}, &hit, 6, {}},
                        {"metal", {synthThud()}, &metal, 4, fs::exists(metalFile) ? metalFile : std::string{}},
                        {"thud", {synthKill(41), synthKill(42)}, &thud, 4, {}},
                        {"hurt", {synthHurt()}, &hurt, 2, {}},
                        {"step", {synthStep(1), synthStep(2), synthStep(3), synthStep(4)}, &step, 4, {}}};

                listener = std::make_unique<AudioListener>();
                attachTo.addRef(*listener);
                for (auto& sp : specs) {
                    std::vector<std::string> paths;
                    if (!sp.file.empty()) {
                        paths.push_back(sp.file);// external sample (e.g. the gun MP3)
                    } else {
                        for (size_t k = 0; k < sp.variants.size(); ++k) {// render the synth fallback(s)
                            auto p = (dir / (sp.name + std::to_string(k) + ".wav")).string();
                            threepp::audio::writeWav(p, sp.variants[k]);
                            paths.push_back(std::move(p));
                        }
                    }
                    for (int i = 0; i < sp.voices; ++i) {
                        auto a = std::make_unique<Audio>(*listener, paths[i % paths.size()]);
                        a->setVolume(0.6f);
                        sp.dst->voices.push_back(std::move(a));
                    }
                }

                // grenade blast: spatialised. Stays at full volume within ~15 m of the
                // camera (covers most of the arena), then a shallow inverse rolloff so
                // far blasts are clearly quieter + directional without dropping out.
                {
                    std::string boomPath = fs::exists(boomFile) ? boomFile : (dir / "boom.wav").string();
                    if (!fs::exists(boomFile)) threepp::audio::writeWav(boomPath, synthBoom());
                    for (int i = 0; i < 4; ++i) {
                        auto a = std::make_unique<PositionalAudio>(*listener, boomPath);
                        a->setVolume(0.9f);// blast is the loudest cue in the game
                        a->setDistanceModel(PositionalAudio::DistanceModel::Inverse);
                        a->setMinDistance(15.f);  // full-volume radius
                        a->setRolloffFactor(0.35f);// shallow falloff beyond it
                        boom.voices.push_back(std::move(a));
                    }
                }
                ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[audio] disabled: " << e.what() << "\n";
            }
        }
    };

    // ========================================================================
    //  SVG HUD toolkit (condensed from examples/loaders/svg_ui.cpp)
    // ========================================================================

    std::shared_ptr<Group> buildSvg(const std::vector<SVGLoader::SVGData>& svgData) {
        auto group = Group::create();
        for (const auto& data : svgData) {
            const auto& fill = data.style.fill;
            if (fill && *fill != "none") {
                auto m = MeshBasicMaterial::create();
                m->color.copy(data.path.color);
                m->opacity = data.style.fillOpacity;
                m->transparent = true;
                m->depthTest = false;
                m->depthWrite = false;
                m->side = Side::Double;
                auto mesh = Mesh::create(ShapeGeometry::create(SVGLoader::createShapes(data)), m);
                mesh->name = data.style.id;
                group->add(mesh);
            }
            const auto& stroke = data.style.stroke;
            if (stroke && *stroke != "none") {
                auto sMat = MeshBasicMaterial::create();
                sMat->color.setStyle(*stroke);
                sMat->opacity = data.style.strokeOpacity;
                sMat->transparent = true;
                sMat->depthTest = false;
                sMat->depthWrite = false;
                sMat->side = Side::Double;
                for (const auto& subPath : data.path.subPaths) {
                    auto sg = SVGLoader::pointsToStroke(subPath->getPoints(), data.style);
                    if (sg) group->add(Mesh::create(sg, sMat));
                }
            }
        }
        return group;
    }

    std::shared_ptr<Group> svgFromString(const std::string& svg) {
        SVGLoader loader;
        return buildSvg(loader.parse(svg));
    }

    std::string hex(int rgb) {
        std::ostringstream os;
        os << '#' << std::hex << std::setfill('0') << std::setw(6) << (rgb & 0xffffff);
        return os.str();
    }

    float wrapPi(float a) {
        while (a > math::PI) a -= 2.f * math::PI;
        while (a < -math::PI) a += 2.f * math::PI;
        return a;
    }

    // Annular wedge (donut slice) path, polyline-sampled — no SVG arc-flag
    // pitfalls. Centred on the origin, opening toward +y (screen-up in the
    // y-up UI overlay), spanning ±halfDeg. rInner = 0 gives a pie slice.
    std::string wedgePath(float rInner, float rOuter, float halfDeg) {
        std::ostringstream d;
        const int N = 18;
        auto ang = [&](int i) { return (90.f - halfDeg + 2.f * halfDeg * i / N) * math::DEG2RAD; };
        for (int i = 0; i <= N; ++i)
            d << (i == 0 ? "M" : "L") << rOuter * std::cos(ang(i)) << "," << rOuter * std::sin(ang(i)) << " ";
        if (rInner > 0.f)
            for (int i = N; i >= 0; --i)
                d << "L" << rInner * std::cos(ang(i)) << "," << rInner * std::sin(ang(i)) << " ";
        else
            d << "L0,0 ";
        d << "Z";
        return d.str();
    }

    // Owned-material rect (so we can recolour / rescale it). Built from <rect>.
    struct RectMesh {
        std::shared_ptr<Mesh> mesh;
        std::shared_ptr<MeshBasicMaterial> material;
    };
    RectMesh rect(float w, float h, int color, float opacity = 1.f) {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width=")" << w
            << R"(" height=")" << h << R"(" fill="#ffffff"/></svg>)";
        SVGLoader loader;
        auto data = loader.parse(svg.str());
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(color);
        mat->opacity = opacity;
        mat->transparent = true;
        mat->depthTest = false;
        mat->depthWrite = false;
        mat->side = Side::Double;
        auto geo = ShapeGeometry::create(SVGLoader::createShapes(data.front()));
        return {Mesh::create(geo, mat), mat};
    }

    // Rounded, border-stroked panel — the HUD framing primitive (<rect rx> + stroke).
    std::shared_ptr<Group> panel(float w, float h, float rx, int fill, float fillOp,
                                 int edge, float edgeW) {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width=")" << w
            << R"(" height=")" << h << R"(" rx=")" << rx << R"(" fill=")" << hex(fill)
            << R"(" fill-opacity=")" << fillOp << R"(" stroke=")" << hex(edge)
            << R"(" stroke-width=")" << edgeW << R"("/></svg>)";
        return svgFromString(svg.str());
    }

    // Collect every MeshBasicMaterial under an svgFromString group, so HUD
    // pieces built from full SVG documents stay restylable per frame.
    std::vector<std::shared_ptr<MeshBasicMaterial>> svgMats(const std::shared_ptr<Group>& g) {
        std::vector<std::shared_ptr<MeshBasicMaterial>> out;
        g->traverseType<Mesh>([&](Mesh& m) {
            if (auto mat = std::dynamic_pointer_cast<MeshBasicMaterial>(m.material())) out.push_back(mat);
        });
        return out;
    }

    std::shared_ptr<TextSprite> makeText(const Font& font, const std::string& text, int color,
                                         float px, float ax, float ay, float ox, float oy,
                                         TextSprite::HorizontalAlignment h = TextSprite::HorizontalAlignment::Left,
                                         TextSprite::VerticalAlignment v = TextSprite::VerticalAlignment::Center) {
        // screen-space sprites bypass the scene graph, so scale px size AND
        // pixel offsets here (uiScale, not a parent group transform)
        auto t = TextSprite::create(font, px * uiScale);
        t->setColor(Color(color));
        t->setText(text);
        t->setHorizontalAlignment(h);
        t->setVerticalAlignment(v);
        t->screenSpace = true;
        t->screenAnchor.set(ax, ay);
        t->position.set(ox * uiScale, oy * uiScale, 0.f);
        return t;
    }

    // A text readout that only re-rasterises when its content changes.
    struct Readout {
        std::shared_ptr<TextSprite> sprite;
        std::string last;
        void set(const std::string& s) {
            if (s == last) return;
            last = s;
            sprite->setText(s);
        }
    };

    // Responsive anchoring: position = anchor*viewport + pixel offset, re-applied
    // on resize. Mirrors the svg_ui layout helper.
    struct Layout {
        std::vector<std::function<void(float, float)>> fns;
        void add(const std::shared_ptr<Object3D>& g, float ax, float ay, float ox, float oy, float z) {
            fns.emplace_back([=](float W, float H) { g->position.set(ax * W + ox * uiScale, ay * H + oy * uiScale, z); });
        }
        void addRaw(std::function<void(float, float)> fn) { fns.emplace_back(std::move(fn)); }
        void apply(float W, float H) {
            for (auto& f : fns) f(W, H);
        }
    };

    // ========================================================================
    //  Game entities
    // ========================================================================

    // One jointed limb of a death ragdoll (spawned when the enemy dies).
    struct RagdollPart {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* body = nullptr;
    };

    struct Enemy {
        std::shared_ptr<Mesh> visual;// capsule body (added to scene)
        std::shared_ptr<MeshStandardMaterial> mat;
        PxRigidDynamic* body = nullptr;
        int hp = kEnemyHp;
        bool alive = true;
        float deadTtl = 0.f;
        float attackCd = 0.f;
        // death ragdoll: the live capsule becomes the torso, these are the limbs
        // jointed to it; both vectors are populated in killEnemy, freed in removeEnemy.
        std::vector<RagdollPart> parts;
        std::vector<PxJoint*> joints;
    };

    struct Ephemeral {// short-lived visual (tracer / flash / spark)
        std::shared_ptr<Object3D> obj;
        float ttl;
    };

}// namespace

int main(int argc, char** argv) {

    // Headless capture (dev): tps_shooter --shot <name.png> [--frames N] —
    // renders N frames from spawn, saves via writeFramebuffer, exits.
    std::string shotPath;
    int shotFrames = 180, shotFrame = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (std::string(argv[i]) == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
    }

#ifndef __APPLE__
    uiScale = monitor::contentScale().first;
#endif
    // scale the window with the HUD, but keep it on screen (a 150% 1080p
    // display has fewer logical pixels than the 1280x800 design)
    const auto screen = monitor::monitorSize();
    const int winW = std::min(static_cast<int>(1280 * uiScale), screen.width() * 9 / 10);
    const int winH = std::min(static_cast<int>(800 * uiScale), screen.height() * 9 / 10);
    Canvas canvas(Canvas::Parameters().title("threepp - Third Person Shooter").size(winW, winH).antialiasing(4));
    // Force the GL backend so the demo launches straight into the game instead
    // of prompting for a renderer on stdin (createRenderer's default behaviour).
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;
    renderer->autoClear = false;
    renderer->toneMapping = ToneMapping::ACESFilmic;// HDR env needs tone mapping
    renderer->toneMappingExposure = 1.0f;

    if (auto pt = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        // pt->setSamplesPerPixel(2);// path trace a few samples per frame to converge over time
        // The hybrid PT anti-aliases via TAA on the raster G-buffer, which the
        // shooter's constant motion keeps rejecting → jaggy edges while moving.
        // Silhouette MSAA fires extra primary rays at detected edge pixels every
        // frame (motion-independent), so crank it from the 8x default to 16x for
        // clean edges; edge pixels are only ~10% of the frame so the cost is local.
        // pt->setSilhouetteMsaaExtra(8);// (N+1)x = 16x MSAA at silhouette pixels
        // pt->setBloomIntensity(1);
        // pt->setSharpenStrength(0.5f);
    }

    // Pointer-lock mouse-look: cursor hidden + grabbed while playing (raw,
    // unbounded deltas), released on game-over so the restart button is
    // clickable, re-grabbed on restart.
    Vector2 lastMouse{-1, -1};
    bool haveMouse = false;
    auto setCursorLocked = [&](bool locked) {
        if (auto* w = canvas.windowPtr()) {
            glfwSetInputMode(w, glfwc::CURSOR, locked ? glfwc::CURSOR_DISABLED : glfwc::CURSOR_NORMAL);
            if (locked && glfwRawMouseMotionSupported())
                glfwSetInputMode(w, glfwc::RAW_MOUSE_MOTION, glfwc::TRUE_);
        }
        haveMouse = false;// skip the first (jumpy) delta after a mode switch
    };

    // ===== world ============================================================
    auto scene = Scene::create();

    auto camera = PerspectiveCamera::create(70, canvas.aspect(), 0.001f, 1000.f);// 1M far/near is fine now: reversed-Z raster (was z-fighting before; near was bumped to 0.1 as a workaround)
    camera->position.set(0, 3, 8);

    // Sun direction for the directional light (raster shadows). On Vulkan the
    // path tracer takes its key light + shadows from the HDR sun instead.
    Vector3 sunDir;
    sunDir.setFromSphericalCoords(1.f, math::degToRad(90.f - 35.f), math::degToRad(55.f));

    // HDR equirectangular environment = the sky AND the image-based light, on
    // every backend. Replaces the procedural Sky shader (a ShaderMaterial the
    // Vulkan path tracer can't render); on Vulkan this is exactly what the PT
    // importance-samples for the visible sky and IBL.
    RGBELoader hdrLoader;
    if (auto hdr = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr")) {
        scene->background = hdr;
        scene->environment = hdr;
    }

    // lighting: the HDR env supplies ambient + IBL; a directional sun adds the
    // crisp shadows the raster backends need (the Vulkan PT shadows off the HDR).
    auto hemi = HemisphereLight::create(0xaeccff, 0x4a4031, 0.25f);
    hemi->position.set(0, 50, 0);
    scene->add(hemi);
    auto sun = DirectionalLight::create(0xffe4bd, 1.8f);
    sun->position.copy(sunDir * 80.f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    sun->shadow->bias = -0.0004f;
    sun->shadow->camera->as<OrthographicCamera>()->left = -kArena;
    sun->shadow->camera->as<OrthographicCamera>()->right = kArena;
    sun->shadow->camera->as<OrthographicCamera>()->top = kArena;
    sun->shadow->camera->as<OrthographicCamera>()->bottom = -kArena;
    sun->shadow->camera->nearPlane = 1.f;
    sun->shadow->camera->farPlane = 240.f;
    sun->shadow->camera->updateProjectionMatrix();
    scene->add(sun);

    // ===== audio ============================================================
    SoundBank sfx;
    sfx.init(*camera);

    // ===== physics ==========================================================
    PhysxWorld world;

    // Maps each PhysX collider back to its visual Mesh so a shot can stamp a
    // decal on whatever it hit (and parent it there so it rides moving crates).
    std::unordered_map<const PxRigidActor*, Mesh*> actorToMesh;

    // ---- materials --------------------------------------------------------
    // Solid PBR colours (no image maps). The GL backend crashes uploading the
    // large non-power-of-two assets (sand.jpg 2070x1381, crate.gif), so the
    // world stays texture-free; the sky + lighting still carry the look.
    auto groundMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xbeae89).roughness(1.f).metalness(0.f));
    auto concreteMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x9a958c).roughness(0.85f).metalness(0.04f));
    auto sandstoneMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xb8a17e).roughness(0.9f).metalness(0.f));
    auto crateMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0xb5793a).roughness(0.7f).metalness(0.05f).map(TextureLoader().load(std::string(DATA_FOLDER) + "/textures/crate.gif")));
    // Kept rough + low envMapIntensity on purpose: glossy metal cylinders expose
    // a vertical-streak seam in the GL backend's cubeUV PMREM IBL (curved surfaces
    // sample the prefiltered-env atlas across a tile boundary; a cylinder's
    // constant-along-axis reflection turns it into a clean streak). WGPU/Vulkan
    // are unaffected — softening the reflection just hides it in the demo.
    auto barrelMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x3f7d4f).roughness(0.7f).metalness(0.2f).envMapIntensity(0.5f));

    // ---- ground ------------------------------------------------------------
    auto ground = Mesh::create(BoxGeometry::create(kArena * 2, 1.f, kArena * 2), groundMat);
    ground->position.y = -0.5f;
    ground->receiveShadow = true;
    scene->add(ground);
    actorToMesh[world.addStatic(*ground)] = ground.get();

    // ---- static-geometry helpers (each collider registers for decals/dust) -
    // Axis-aligned or rotated box. Shape is inferred from the geometry (no
    // scaling), so ramps tilt via `pitch` rather than a scaled mesh.
    auto addBox = [&](const Vector3& pos, const Vector3& size, const std::shared_ptr<Material>& mat, float yaw = 0.f, float pitch = 0.f) {
        auto m = Mesh::create(BoxGeometry::create(size.x, size.y, size.z), mat);
        m->position.copy(pos);
        m->rotation.set(pitch, yaw, 0.f);
        m->castShadow = true;
        m->receiveShadow = true;
        scene->add(m);
        actorToMesh[world.addStatic(*m)] = m.get();
        return m;
    };
    auto addPillar = [&](float x, float z, float radius, float height, const std::shared_ptr<Material>& mat) {
        auto m = Mesh::create(CylinderGeometry::create(radius, radius, height, 20), mat);
        m->position.set(x, height * 0.5f, z);
        m->castShadow = true;
        m->receiveShadow = true;
        scene->add(m);
        if (auto* b = world.addStaticTrimesh(*m)) actorToMesh[b] = m.get();
        return m;
    };

    // ---- perimeter walls + corner towers ----------------------------------
    const float wallH = 5.f;
    addBox({0, wallH * 0.5f, -kArena}, {kArena * 2, wallH, 1.f}, concreteMat);
    addBox({0, wallH * 0.5f, kArena}, {kArena * 2, wallH, 1.f}, concreteMat);
    addBox({-kArena, wallH * 0.5f, 0}, {1.f, wallH, kArena * 2}, concreteMat);
    addBox({kArena, wallH * 0.5f, 0}, {1.f, wallH, kArena * 2}, concreteMat);
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            addBox({sx * kArena, 3.5f, sz * kArena}, {4.5f, 7.f, 4.5f}, sandstoneMat);

    // ---- central pillar plaza ---------------------------------------------
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sz = -1; sz <= 1; sz += 2)
            addPillar(sx * 8.f, sz * 8.f, 0.55f, 6.f, sandstoneMat);
    (void) addPillar;

    // ---- raised bunker with a ramp up (rewards the jump) ------------------
    {
        const float pz = -16.f, pw = 12.f, pd = 7.f, ph = 1.4f;
        addBox({0, ph * 0.5f, pz}, {pw, ph, pd}, concreteMat);                      // platform (top at ph; low enough to jump onto the edge)
        addBox({0, ph + 0.5f, pz - pd * 0.5f + 0.5f}, {pw, 1.f, 1.f}, sandstoneMat);// rear parapet (cover up top)
        // ramp on the +Z (centre-facing) edge. Flip the pitch sign if it tilts
        // the wrong way.
        const float run = 7.f;
        addBox({0, ph * 0.5f - 0.1f, pz + pd * 0.5f + run * 0.5f - 1.f}, {5.f, 0.5f, run}, concreteMat, 0.f, std::atan2(ph, run));
    }

    // ---- corner L-cover ----------------------------------------------------
    auto addCornerCover = [&](float cx, float cz) {
        const float ix = cx > 0 ? -1.f : 1.f, iz = cz > 0 ? -1.f : 1.f;
        addBox({cx + ix * 2.f, 0.8f, cz}, {5.f, 1.6f, 1.f}, sandstoneMat);
        addBox({cx, 0.8f, cz + iz * 2.f}, {1.f, 1.6f, 5.f}, sandstoneMat);
    };
    addCornerCover(-18, -18);
    addCornerCover(18, -18);
    addCornerCover(-18, 18);
    addCornerCover(18, 18);

    // ---- mid-lane sandbag cover -------------------------------------------
    addBox({17, 0.5f, 0}, {1.2f, 1.f, 6.f}, sandstoneMat);
    addBox({-17, 0.5f, 0}, {1.2f, 1.f, 6.f}, sandstoneMat);
    addBox({0, 0.5f, 17}, {6.f, 1.f, 1.2f}, sandstoneMat);

    // ---- dynamic crates (shootable / pushable) — home pose kept for restart -
    struct Dynamic {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* body;
        Vector3 home;
    };
    std::vector<Dynamic> dynamics;
    auto crateGeo = BoxGeometry::create(1.f, 1.f, 1.f);
    auto spawnCrateStack = [&](float cx, float cz, int height) {
        for (int y = 0; y < height; ++y) {
            auto m = Mesh::create(crateGeo, crateMat);
            m->position.set(cx + frand(-0.05f, 0.05f), 0.5f + y * 1.001f, cz + frand(-0.05f, 0.05f));
            m->castShadow = true;
            m->receiveShadow = true;
            scene->add(m);
            auto* body = world.add(*m, 60.f);
            actorToMesh[body] = m.get();
            dynamics.push_back({m, body, m->position.clone()});
        }
    };
    spawnCrateStack(13, 11, 3);
    spawnCrateStack(-13, 12, 2);
    spawnCrateStack(12, -12, 3);
    spawnCrateStack(-12, -13, 2);
    spawnCrateStack(6, -14, 2);

    // ---- dynamic barrels (convex collider cooked from cylinder verts) ------
    auto barrelGeo = CylinderGeometry::create(0.45f, 0.45f, 1.2f, 18);
    auto spawnBarrels = [&](float cx, float cz, int n) {
        for (int i = 0; i < n; ++i) {
            auto m = Mesh::create(barrelGeo, barrelMat);
            m->position.set(cx + frand(-1.2f, 1.2f), 0.6f, cz + frand(-1.2f, 1.2f));
            m->castShadow = true;
            m->receiveShadow = true;
            scene->add(m);
            if (auto* body = world.addDynamicConvex(*m, 35.f)) {
                actorToMesh[body] = m.get();
                dynamics.push_back({m, body, m->position.clone()});
            }
        }
    };
    spawnBarrels(16, -4, 3);
    spawnBarrels(-16, 4, 3);
    spawnBarrels(4, 16, 2);

    // ===== enemy navigation grid (flow field) ==============================
    // A coarse occupancy grid over the arena. Each frame the player has moved to
    // a new cell, a BFS distance field is flooded out from the player's cell, so
    // every reachable free cell knows its step-count to the player. Enemies steer
    // DOWN that gradient — which means they take real paths around cover instead
    // of grinding into it. STATIC geometry (walls / towers / pillars / cover)
    // blocks cells; the dynamic crate + barrel stacks also mark their resting
    // footprint so bots skirt them. Bot-vs-bot crowding is handled by a local
    // separation nudge in the AI loop below.
    const int gridN = static_cast<int>(std::lround(kArena * 2.f / kNavCell));
    auto navIdx = [&](int r, int c) { return static_cast<size_t>(r) * gridN + c; };
    auto colOf = [&](float w) { return std::clamp(static_cast<int>((w + kArena) / kNavCell), 0, gridN - 1); };
    auto cellCenter = [&](int c) { return -kArena + (static_cast<float>(c) + 0.5f) * kNavCell; };
    std::vector<uint8_t> navBlocked(static_cast<size_t>(gridN) * gridN, 0);
    {
        PxOverlapBuffer ob;
        const PxBoxGeometry probe(kNavCell * 0.45f, 0.7f, kNavCell * 0.45f);
        PxQueryFilterData fd;
        fd.flags = PxQueryFlags(PxQueryFlag::eSTATIC);// only static geometry blocks the nav grid
        for (int gz = 0; gz < gridN; ++gz)
            for (int gx = 0; gx < gridN; ++gx) {
                const PxTransform pose(PxVec3(cellCenter(gx), 0.9f, cellCenter(gz)));
                if (world.scene().overlap(probe, pose, ob, fd)) navBlocked[navIdx(gz, gx)] = 1;
            }
        for (auto& dyn : dynamics) navBlocked[navIdx(colOf(dyn.home.z), colOf(dyn.home.x))] = 1;
    }
    std::vector<int> navDist(static_cast<size_t>(gridN) * gridN, -1);
    std::vector<int> navFrontier;
    int navPlayerRow = -1, navPlayerCol = -1;
    auto rebuildFlow = [&](int prow, int pcol) {
        std::fill(navDist.begin(), navDist.end(), -1);
        navFrontier.clear();
        navDist[navIdx(prow, pcol)] = 0;
        navFrontier.push_back(prow * gridN + pcol);
        size_t head = 0;
        while (head < navFrontier.size()) {
            const int cur = navFrontier[head++];
            const int r = cur / gridN, c = cur % gridN;
            const int nd = navDist[navIdx(r, c)] + 1;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    if (!dr && !dc) continue;
                    const int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= gridN || nc < 0 || nc >= gridN) continue;
                    if (navBlocked[navIdx(nr, nc)] || navDist[navIdx(nr, nc)] != -1) continue;
                    if (dr && dc && (navBlocked[navIdx(r, nc)] || navBlocked[navIdx(nr, c)])) continue;// no corner-cut
                    navDist[navIdx(nr, nc)] = nd;
                    navFrontier.push_back(nr * gridN + nc);
                }
        }
    };

    // ===== player ===========================================================
    // Collision proxy: a capsule rigid body with locked rotation, driven by
    // setting horizontal velocity each frame (classic kinematic-velocity
    // character) while PhysX handles gravity, walls and slopes.
    auto playerProxy = Mesh::create(CapsuleGeometry::create(kPlayerRadius, kPlayerLen), MeshBasicMaterial::create());
    playerProxy->visible = false;
    playerProxy->position.set(0, kPlayerHalf, 0);
    auto* playerBody = world.add(*playerProxy, 80.f);
    playerBody->setRigidDynamicLockFlags(
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
            PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
    playerBody->setMaxLinearVelocity(40.f);
    {
        // Disable scene queries on the player's own shape so our aim/ground
        // raycasts never hit the player (the camera sits behind them).
        PxShape* sh = nullptr;
        playerBody->getShapes(&sh, 1);
        if (sh) sh->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, false);
    }

    // Visual: the SWAT character + Mixamo rifle clip set (assets/swat.glb).
    auto playerRig = Group::create();
    scene->add(playerRig);
    std::unique_ptr<AnimationMixer> mixer;

    // Named clips resolved to actions. Locomotion clips loop; one-shots play once.
    struct PlayerAnims {
        AnimationAction* idle = nullptr;    // rifle aiming idle
        AnimationAction* walk = nullptr;    // walking
        AnimationAction* walkBack = nullptr;// walking backwards
        AnimationAction* run = nullptr;     // rifle run
        AnimationAction* runBack = nullptr; // run backwards
        AnimationAction* strafeL = nullptr; // strafe left
        AnimationAction* strafeR = nullptr; // strafe right
        AnimationAction* fire = nullptr;    // firing rifle
        AnimationAction* reload = nullptr;  // reloading
        AnimationAction* jump = nullptr;    // rifle jump
        AnimationAction* hit = nullptr;     // hit reaction
        AnimationAction* grenade = nullptr; // toss grenade (additive overlay)
    } pa;
    AnimationAction* currentA = nullptr;
    Object3D* handBone = nullptr;
    Object3D* leftHandBone = nullptr;// left hand -> rifle fore-grip (defines the barrel line)
    Object3D* spineBone = nullptr;   // upper-spine bone tilted to aim the gun up/down
    Object3D* hipsBone = nullptr;
    Vector3 hipsBind;// Hips bind-pose local position; clip root-motion (X/Z) is pinned to this
    {
        GLTFLoader loader;
        const std::string swatPath = std::string(DATA_FOLDER) + "/models/gltf/swat.glb";
        auto res = loader.load(swatPath);
        if (res) {
            auto& model = res->scene;
            // Skinned meshes: the geometry bounding sphere is the BIND pose, not the
            // animated/scaled world pose, so frustum culling wrongly drops the body
            // when the ADS camera pulls in close with a narrow FOV (the gun is a
            // separate static mesh with correct bounds, so it stayed). The player is
            // always on-screen and relevant — just never cull it.
            model->traverseType<Mesh>([](Mesh& m) {
                m.castShadow = true;
                m.frustumCulled = false;
            });

            // Normalise scale to the capsule. NB: Box3::setFromObject measures the
            // *un-skinned* geometry box, which on this rig is tiny — the skeleton's
            // bind matrices carry the real scale. So measure the skeleton's world
            // vertical span (bone positions reflect the skinned size) instead.
            model->updateMatrixWorld(true);
            float minY = 1e9f, maxY = -1e9f;
            model->traverse([&](Object3D& o) {
                Vector3 wp;
                wp.setFromMatrixPosition(*o.matrixWorld);
                minY = std::min(minY, wp.y);
                maxY = std::max(maxY, wp.y);
            });
            const float skelH = maxY - minY;
            const float modelH = skelH > 1e-4f ? skelH : 1.f;
            model->scale *= kCharHeight / modelH;// skeleton span ≈ standing height
            playerRig->add(model);

            mixer = std::make_unique<AnimationMixer>(*model);
            // Exact (case-insensitive) clip lookup — names come straight from the
            // Mixamo files (see scripts/mixamo_to_glb.py output).
            auto pick = [&](std::string want) -> AnimationAction* {
                std::transform(want.begin(), want.end(), want.begin(), ::tolower);
                for (auto& c : res->animations) {
                    std::string n = c->name();
                    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                    if (n == want) return mixer->clipAction(c);
                }
                return nullptr;
            };
            // Reload + fire are UPPER-BODY OVERLAYS: convert them to additive clips
            // (deltas vs their first frame) so they layer on top of the locomotion
            // base — the legs keep walking while the arms reload/fire. Must run
            // BEFORE clipAction so the action inherits the Additive blend mode.
            for (auto& c : res->animations) {
                std::string n = c->name();
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (n == "reloading" || n == "firing rifle" || n == "toss grenade") c->makeAdditive();
            }
            pa.idle = pick("rifle aiming idle");
            pa.walk = pick("walking");
            pa.walkBack = pick("walking backwards");
            pa.run = pick("rifle run");
            pa.runBack = pick("run backwards");
            pa.strafeL = pick("strafe left");
            pa.strafeR = pick("strafe right");
            pa.fire = pick("firing rifle");
            pa.reload = pick("reloading");
            pa.jump = pick("rifle jump");
            pa.hit = pick("hit reaction");
            pa.grenade = pick("toss grenade");

            // Everything loops — including the reload/hit overlays. A Loop::Once
            // clip sets enabled=false when it finishes, which drops it to weight 0;
            // while it's still the active clip (e.g. auto-reload after emptying the
            // mag) the skeleton then falls to its bind pose (T-pose). We crossfade
            // these overlays out by game state, so looping is harmless.
            for (auto* a : {pa.idle, pa.walk, pa.walkBack, pa.run, pa.runBack,
                            pa.strafeL, pa.strafeR, pa.fire, pa.jump, pa.reload, pa.hit, pa.grenade})
                if (a) a->setLoop(Loop::Repeat);
            if (pa.reload) pa.reload->setDuration(kReloadTime); // one play-through ≈ reload time
            if (pa.grenade) pa.grenade->setDuration(kThrowTime);// one throw spans the cooldown
            // Stretch the jump clip to span the airtime so it plays ONCE (it's
            // shorter than the ~1s jump, so at natural speed it loops mid-air =
            // "plays twice"). ~2*kJumpSpeed/gravity; tune if the jump feels floaty.
            if (pa.jump) pa.jump->setDuration(1.1f);

            if (!pa.idle && !res->animations.empty()) pa.idle = mixer->clipAction(res->animations.front());
            currentA = pa.idle;
            if (currentA) currentA->play();

            // Additive overlays stay active but silent; their weight is driven each
            // frame (0 = off, 1 = full) so they blend onto whatever the base plays.
            if (pa.reload) {
                pa.reload->play();
                pa.reload->setEffectiveWeight(0.f);
            }
            if (pa.fire) {
                pa.fire->play();
                pa.fire->setEffectiveWeight(0.f);
            }
            if (pa.grenade) {
                pa.grenade->play();
                pa.grenade->setEffectiveWeight(0.f);
            }

            // Right-hand bone (rifle attach) + Hips/root bone (root-motion pin).
            model->traverse([&](Object3D& o) {
                if (o.name.empty()) return;
                std::string n = o.name;
                std::transform(n.begin(), n.end(), n.begin(), ::tolower);
                if (!handBone && n.find("righthand") != std::string::npos) handBone = &o;
                if (!leftHandBone && n.find("lefthand") != std::string::npos) leftHandBone = &o;
                // upper spine for the aim tilt; prefer spine2 (chest) over spine/spine1
                if (n.find("spine") != std::string::npos) {
                    if (n.find("spine2") != std::string::npos) spineBone = &o;
                    else if (!spineBone) spineBone = &o;
                }
                if (!hipsBone && n.find("hips") != std::string::npos) {
                    hipsBone = &o;
                    hipsBind = o.position;// bind-pose Hips local position
                }
            });

            std::cout << "Loaded SWAT player: " << res->animations.size() << " clip(s); height "
                      << modelH << " -> " << (2.f * kPlayerHalf) << "m; bones R-hand="
                      << (handBone ? "ok" : "MISSING") << " L-hand=" << (leftHandBone ? "ok" : "MISSING")
                      << " spine=" << (spineBone ? "ok" : "MISSING") << std::endl;
        } else {
            std::cerr << "Failed to load " << swatPath << "\n";
        }
    }

    // ---- rifle model, pinned into the right hand each frame ----------------
    // Mixamo's rifle clips pose the hands around a weapon but ship none, so we
    // load the rifle GLB and drive its world transform from the hand bone
    // (decoupled from the skeleton scale). The four constants are tweak-to-taste
    // — verify in-hand after a build and nudge.
    const Vector3 kRifleGripPos{0.f, 0.15f, 0.f};// offset from hand bone (m)
    const float kRifleLen = 0.72f;               // normalised longest-axis length (m)
    const Euler kRifleModelRot{0.f, 0.f, 0.f};   // model is already barrel:+Z / up:+Y
    const float kRifleUpRotDeg = -25.f;          // up-axis spin for the two-handed hold (clockwise)
    const float kRifleUpRotReloadDeg = 5.f;     // up-axis spin while reloading/throwing (counter-clockwise)
    Vector3 muzzleLocal{0.f, 0.f, 0.f};          // barrel-tip offset in rifle-local space (set at load)
    auto rifle = Group::create();
    {
        GLTFLoader gloader;
        if (auto r = gloader.load(std::string(DATA_FOLDER) + "/models/gltf/rifle.glb")) {
            auto& gun = r->scene;
            gun->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
            gun->updateMatrixWorld(true);
            Box3 gb;// a static mesh, so the geometry box is correct (no skinning)
            gb.setFromObject(*gun);
            const Vector3 gsz = gb.getSize();
            const float gmax = std::max({gsz.x, gsz.y, gsz.z});
            if (gmax > 1e-4f) gun->scale *= kRifleLen / gmax;
            gun->rotation.set(kRifleModelRot.x, kRifleModelRot.y, kRifleModelRot.z);
            rifle->add(gun);
            // Barrel tip in rifle-local space = +Z-face centre of the scaled gun box.
            // The rifle Group is still at the origin/identity here, so the gun's world
            // box equals its rifle-local box. Used as the tracer / muzzle-flash origin.
            rifle->updateMatrixWorld(true);
            Box3 lb;
            lb.setFromObject(*gun);
            const Vector3 lc = lb.getCenter();
            muzzleLocal.set(lc.x, lc.y, lb.max().z);
            std::cout << "Loaded rifle.glb (native " << gsz.x << " x " << gsz.y << " x " << gsz.z << ")" << std::endl;
        } else {
            std::cerr << "Failed to load rifle.glb\n";
        }
        rifle->visible = (handBone != nullptr);
        scene->add(rifle);
    }

    // ===== camera rig (third person, mouse look) ============================
    float camYaw = 0.f, camPitch = 0.45f, camDist = 4.5f;
    float recoilKick = 0.f;// transient upward aim kick from firing, recovers to 0
    float recoilYaw = 0.f; // transient horizontal aim jitter from firing
    bool aiming = false;        // RMB held -> aim-down-sights zoom
    float zoomT = 0.f;          // 0 = hip, 1 = aiming (eased each frame)
    bool inspect = false;       // middle-mouse held -> swing camera to face the player
    float inspectT = 0.f;       // 0 = behind, 1 = facing the player (eased each frame)
    float camBoom = camDist;    // current collision-clamped camera distance (snaps in, eases out)
    Vector3 aimDir(0.f, 0.f, 1.f);// camera forward; where shots + grenades go (matches the crosshair)
    Vector3 playerPos{0, kPlayerHalf, 0};

    // landing-edge detection (jump-clip → locomotion transition)
    bool wasGrounded = true;

    // ===== enemies ==========================================================
    std::vector<std::unique_ptr<Enemy>> enemies;
    auto enemyGeo = CapsuleGeometry::create(0.35f, 1.0f);
    auto headGeo = SphereGeometry::create(0.22f, 16, 12);
    auto armGeo = CapsuleGeometry::create(0.08f, 0.45f);// ragdoll arms
    auto legGeo = CapsuleGeometry::create(0.10f, 0.55f);// ragdoll legs
    float enemySpawnTimer = 0.f;

    auto spawnEnemy = [&]() {
        const float ang = frand(0.f, 2.f * math::PI);
        const float r = frand(kArena * 0.5f, kArena - 3.f);
        auto e = std::make_unique<Enemy>();
        e->mat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(0xc0392b).roughness(0.65f).metalness(0.f).envMapIntensity(0.6f));
        e->visual = Mesh::create(enemyGeo, e->mat);
        e->visual->position.set(std::cos(ang) * r, 0.9f, std::sin(ang) * r);
        e->visual->castShadow = true;
        auto head = Mesh::create(headGeo, MeshStandardMaterial::create(
                                                  MeshStandardMaterial::Params{}.color(0x2c2c2c).roughness(0.5f)));
        head->position.set(0, 0.62f, 0.05f);
        e->visual->add(head);
        scene->add(e->visual);
        e->body = world.add(*e->visual, 45.f);
        e->body->setRigidDynamicLockFlags(
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_X |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y |
                PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z);
        Enemy* raw = e.get();
        e->body->userData = raw;
        enemies.push_back(std::move(e));
    };

    // Death = ragdoll. The live capsule becomes the torso (angular locks lifted so
    // it tumbles); four thin limb bodies are spawned at the shoulders/hips and tied
    // on with limited spherical joints so the corpse flops believably. The kill
    // impulse flings the torso; the limbs inherit a softened, jittered share.
    auto killEnemy = [&](Enemy* e, const Vector3& impulseDir) {
        e->alive = false;
        e->deadTtl = kRagdollTtl;
        e->mat->color = Color(0x3a1512);
        e->body->setRigidDynamicLockFlags(PxRigidDynamicLockFlags(0));// free to tumble
        e->body->setAngularDamping(0.6f);
        e->body->setLinearDamping(0.05f);

        const PxTransform tp = e->body->getGlobalPose();
        const Quaternion torsoQ = fromPxQuat(tp.q);

        // joint frame whose +X (the spherical-cone axis) points along `axisLocal`
        auto coneFrame = [](const Vector3& anchorLocal, const Vector3& axisLocal) {
            Quaternion q;
            q.setFromUnitVectors(Vector3(1, 0, 0), axisLocal.clone().normalize());
            return PxTransform(toPxVec3(anchorLocal), toPxQuat(q));
        };

        // attach a limb capsule: top end pinned to `shoulderLocal` on the torso,
        // hanging along `dirLocal` (torso-local), constrained to a swing cone.
        auto addLimb = [&](const std::shared_ptr<CapsuleGeometry>& geo, float radius, float len,
                           const Vector3& shoulderLocal, const Vector3& dirLocal) {
            const Vector3 shoulderW = fromPxVec3(tp.transform(toPxVec3(shoulderLocal)));
            Vector3 dirW = dirLocal.clone().applyQuaternion(torsoQ);
            dirW.normalize();
            // limb-local +Y runs from the free (bottom) end up to the anchor; so it
            // maps to -dirW in world. Centre sits half a length below the shoulder.
            Quaternion limbQ;
            limbQ.setFromUnitVectors(Vector3(0, 1, 0), dirW.clone().multiplyScalar(-1.f));
            auto lm = Mesh::create(geo, e->mat);
            lm->castShadow = true;
            lm->position.copy(shoulderW + dirW * (len * 0.5f));
            lm->quaternion.copy(limbQ);
            scene->add(lm);
            auto* lb = world.add(*lm, 26.f);
            if (!lb) {
                scene->remove(*lm);
                return;
            }
            lb->setAngularDamping(0.6f);
            lb->setLinearDamping(0.05f);
            auto* j = PxSphericalJointCreate(
                    world.physics(),
                    e->body, coneFrame(shoulderLocal, dirLocal),
                    lb, coneFrame(Vector3(0, len * 0.5f, 0), Vector3(0, -1, 0)));
            j->setLimitCone(PxJointLimitCone(PxPi * 0.40f, PxPi * 0.40f));
            j->setSphericalJointFlag(PxSphericalJointFlag::eLIMIT_ENABLED, true);
            e->parts.push_back({lm, lb});
            e->joints.push_back(j);
        };

        // shoulders near the capsule top, hips near the bottom (capsule half-len 0.5)
        addLimb(armGeo, 0.08f, 0.45f, {0.40f, 0.40f, 0.f}, {0.45f, -1.f, 0.f});
        addLimb(armGeo, 0.08f, 0.45f, {-0.40f, 0.40f, 0.f}, {-0.45f, -1.f, 0.f});
        addLimb(legGeo, 0.10f, 0.55f, {0.16f, -0.50f, 0.f}, {0.18f, -1.f, 0.f});
        addLimb(legGeo, 0.10f, 0.55f, {-0.16f, -0.50f, 0.f}, {-0.18f, -1.f, 0.f});

        PxRigidBodyExt::addForceAtPos(*e->body,
                                      toPxVec3((impulseDir + Vector3(0, 0.7f, 0)) * 36.f),
                                      tp.p + PxVec3(0, 0.4f, 0),
                                      PxForceMode::eIMPULSE);
        for (auto& p : e->parts) {
            p.body->addForce(toPxVec3((impulseDir + Vector3(frand(-0.3f, 0.3f), 0.5f, frand(-0.3f, 0.3f))) * 5.f),
                             PxForceMode::eIMPULSE);
        }
    };

    auto removeEnemy = [&](Enemy* e) {
        // joints must be released before the actors they constrain
        for (auto* j : e->joints) j->release();
        e->joints.clear();
        for (auto& p : e->parts) {
            world.unbind(*p.mesh);
            world.scene().removeActor(*p.body);
            p.body->release();
            scene->remove(*p.mesh);
        }
        e->parts.clear();
        world.unbind(*e->visual);
        world.scene().removeActor(*e->body);
        e->body->release();
        scene->remove(*e->visual);
    };

    // ===== transient effects (tracers / flashes / sparks) ===================
    std::vector<Ephemeral> fx;
    auto tracerMat = LineBasicMaterial::create();
    tracerMat->color = Color(0xfff2a0);
    tracerMat->transparent = true;
    tracerMat->depthWrite = false;
    auto flashGeo = SphereGeometry::create(0.16f, 8, 6);

    // ===== ejected shell casings ============================================
    // Spent brass tossed from the breech on each shot. Simulated as cheap
    // ballistic props (gravity + tumble + a damped ground bounce, TTL-recycled) —
    // NOT physics bodies, since spawning a rigid actor ~9×/s would churn the
    // scene + acceleration structures.
    auto casingGeo = CylinderGeometry::create(0.011f, 0.011f, 0.05f, 8);
    auto casingMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(0xc8a24a).roughness(0.4f).metalness(0.8f).envMapIntensity(0.5f));
    struct Casing {
        std::shared_ptr<Mesh> mesh;
        Vector3 vel;
        Vector3 spinAxis;
        float spinRate = 0.f;
        float ttl = 0.f;
    };
    std::vector<Casing> casings;
    const size_t kMaxCasings = 24;      // recycle the oldest beyond this
    const float kCasingEjectSide = -1.f; // which side the brass ejects from

    // ===== game state =======================================================
    int health = 100;
    int ammo = kMagSize;
    int score = 0;
    bool reloading = false;
    float reloadTimer = 0.f;
    float fireTimer = 0.f;
    bool firing = false;    // LMB held
    bool firedEmpty = false;// debounce empty click
    bool gameOver = false;
    float hitMarkerT = 0.f;// >0 while the hit marker flashes
    bool  hitWasKill = false;// last hit was a kill → red, popped marker + score float
    float scorePopT  = 0.f;  // >0 while the "+100" kill pop floats up
    float chipHealth = 100.f;// damage-lag bar: eases down toward `health`
    float chSpread   = 6.f;  // crosshair tick spread (eased toward dynamic target)
    float hudT       = 0.f;  // HUD wall-clock for pulses/sweeps
    // Damage-direction arcs: a small pool of wedge overlays around the
    // crosshair; each records the attacker's world bearing and fades out.
    struct DmgArc {
        std::shared_ptr<Group> g;
        std::vector<std::shared_ptr<MeshBasicMaterial>> mats;
        float t = 0.f;
        float bearing = 0.f;
    };
    std::array<DmgArc, 3> dmgArcs;
    int dmgArcNext = 0;
    float stepTimer = 0.f;
    bool stepLeft = false;           // alternates per footstep for the L/R pitch split
    float hitReactTimer = 0.f;       // >0 while the hit-reaction clip plays
    float inFwd = 0.f, inStr = 0.f;  // player's intended move dir (drives anim choice)
    float reloadW = 0.f, fireW = 0.f;// smoothed weights of the additive reload/fire overlays
    bool wasReloading = false;       // rising-edge detect to restart the reload overlay

    // ===== bullet-impact decals =============================================
    // Reuses threepp's DecalGeometry: each shot projects a scorch decal onto the
    // hit surface (the textured-splat decal asset, tinted dark), parented to the
    // hit mesh so it rides knocked-around crates/barrels.
    TextureLoader texLoader;
    auto decalMat = MeshPhongMaterial::create();
    decalMat->map = texLoader.load(std::string(DATA_FOLDER) + "/textures/decal/decal-diffuse.png", ColorSpace::sRGB);
    decalMat->normalMap = texLoader.load(std::string(DATA_FOLDER) + "/textures/decal/decal-normal.jpg", ColorSpace::NoColorSpace);
    decalMat->normalScale.set(1.f, 1.f);
    decalMat->color = Color(0x161616);// scorch tint
    decalMat->specular = Color(0x222222);
    decalMat->shininess = 18.f;
    decalMat->transparent = true;
    decalMat->depthTest = true;
    decalMat->depthWrite = false;
    decalMat->polygonOffset = true;// lift off the surface to beat z-fighting
    decalMat->polygonOffsetFactor = -4.f;

    std::vector<std::shared_ptr<Mesh>> decals;
    auto stampDecal = [&](Mesh* target, const Vector3& point, const Vector3& worldNormal) {
        if (!target) return;
        target->updateMatrixWorld();

        // Orientation: look from the hit point along the surface normal (same
        // recipe as examples/objects/decal.cpp), plus a random roll for variety.
        Matrix4 helper;
        helper.setPosition(point);
        helper.lookAt(point, point + worldNormal, Vector3::Z());
        Euler orientation;
        orientation.setFromRotationMatrix(helper);
        orientation.z = frand(0.f, math::PI * 2.f);

        const float s = frand(0.26f, 0.40f);
        auto geo = DecalGeometry::create(*target, point, orientation, Vector3(s, s, s));
        // Coplanar with the surface; the decal material's polygonOffset keeps it
        // from z-fighting (honored by both the GL and WGPU backends).
        auto decal = Mesh::create(geo, decalMat);

        // DecalGeometry bakes vertices in WORLD space. Parent to the hit mesh
        // with a local matrix of inverse(target world) so the decal stays glued
        // when the target moves (identity for static surfaces).
        Matrix4 inv;
        inv.copy(*target->matrixWorld).invert();
        decal->matrixAutoUpdate = false;
        decal->matrix->copy(inv);
        decal->matrixWorldNeedsUpdate = true;
        target->add(decal);

        decals.push_back(decal);
        if (static_cast<int>(decals.size()) > kMaxDecals) {
            decals.front()->removeFromParent();
            decals.erase(decals.begin());
        }
    };

    // ===== impact particles (dust on surfaces, blood on enemies) ============
    // A short-lived burst of camera-facing billboard sprites per hit, integrated
    // on the CPU (velocity + gravity + drag) and faded via material opacity.
    // NB: NOT Points — PointsMaterial renders at 1px on the WGPU backend (no
    // gl_PointSize), so world-sized points are invisible there. Sprites are
    // world-sized billboards on every backend. Soft round look from the
    // smoke / disc sprite textures.
    auto dustTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png", ColorSpace::sRGB);
    auto bloodTex = texLoader.load(std::string(DATA_FOLDER) + "/textures/sprites/disc.png", ColorSpace::sRGB);

    struct ParticleBurst {
        std::shared_ptr<Group> group;       // holds the billboards
        std::shared_ptr<SpriteMaterial> mat;// shared, faded together
        std::vector<std::shared_ptr<Sprite>> sprites;
        std::vector<Vector3> pos, vel;
        float ttl, life, gravity, drag;
    };
    std::vector<ParticleBurst> bursts;

    auto spawnParticles = [&](const Vector3& point, const Vector3& dir, const Vector3& normal, bool blood) {
        const int count = blood ? 14 : 11;
        ParticleBurst b;
        b.pos.resize(count);
        b.vel.resize(count);

        // Blood sprays back toward the shooter and out of the surface; dust
        // puffs outward along the surface normal.
        Vector3 base = blood ? (normal * 0.5f - dir * 0.9f) : normal;
        if (base.length() < 1e-4f) base = normal;
        base.normalize();

        b.mat = SpriteMaterial::create();
        b.mat->map = blood ? bloodTex : dustTex;
        b.mat->color = blood ? Color(0x9a0d0d) : Color(0x9c8f76);
        b.mat->transparent = true;
        b.mat->depthWrite = false;
        const float size = blood ? 0.16f : 0.28f;

        b.group = Group::create();
        for (int i = 0; i < count; ++i) {
            b.pos[i] = point;
            Vector3 v = base + Vector3(frand(-1.f, 1.f), frand(-1.f, 1.f), frand(-1.f, 1.f)) * (blood ? 0.45f : 0.75f);
            if (v.length() < 1e-4f) v = base;
            v.normalize();
            b.vel[i] = v * (blood ? frand(2.5f, 6.5f) : frand(1.2f, 3.2f));
            auto s = Sprite::create(b.mat);
            s->position.copy(point);
            s->scale.set(size, size, 1.f);
            b.group->add(s);
            b.sprites.push_back(s);
        }
        scene->add(b.group);
        b.life = b.ttl = blood ? 0.55f : 0.70f;
        b.gravity = blood ? 16.f : 3.5f;
        b.drag = blood ? 1.5f : 3.0f;
        bursts.push_back(std::move(b));
    };

    // ===== fire (PhysX raycast) =============================================
    auto fire = [&]() {
        sfx.shot.play(1.f, frand(0.97f, 1.03f));// slight per-shot pitch drift de-loops rapid fire
        ammo--;
        fireTimer = kFireInterval;

        // recoil: kick the aim up + a touch sideways; both recover between shots
        recoilKick = std::min(recoilKick + kRecoilPerShot, kRecoilMax);
        recoilYaw += frand(-kRecoilYawKick, kRecoilYawKick);

        Vector3 origin = camera->position;
        Vector3 dir = aimDir;// camera forward — matches the centre crosshair (set in the camera rig)

        PxRaycastBuffer hitBuf;
        const bool hasHit = world.scene().raycast(toPxVec3(origin), toPxVec3(dir), 200.f, hitBuf) && hitBuf.hasBlock;

        // tracer / muzzle flash start at the actual barrel tip (the gun is oriented
        // to aim along `dir`, so the tip sits on the shot line). The raycast above
        // still originates at the camera so the centre crosshair stays pixel-accurate.
        Vector3 muzzle;
        if (handBone && rifle->visible) {
            rifle->updateMatrixWorld(true);
            muzzle.copy(muzzleLocal).applyMatrix4(*rifle->matrixWorld);
        } else {
            muzzle = playerPos + Vector3(0, kPlayerLen * 0.9f, 0) + dir * 0.6f;
        }
        Vector3 end = hasHit ? fromPxVec3(hitBuf.block.position) : origin + dir * 200.f;

        // tracer
        auto tg = BufferGeometry::create();
        tg->setFromPoints(std::vector<Vector3>{muzzle, end});
        auto tracer = Line::create(tg, tracerMat);
        scene->add(tracer);
        fx.push_back({tracer, 0.05f});

        // muzzle flash
        auto flashMat = MeshBasicMaterial::create();
        flashMat->color = Color(0xffe08a);
        flashMat->transparent = true;
        auto flash = Mesh::create(flashGeo, flashMat);
        flash->position.copy(muzzle);
        scene->add(flash);
        fx.push_back({flash, 0.05f});

        // eject a spent casing from the breech (right side of the gun), arcing out
        if (handBone && rifle->visible) {
            // rifle world basis (parent = scene identity, so its quaternion is world)
            Vector3 fwd(0.f, 0.f, 1.f), rightV(1.f, 0.f, 0.f), up(0.f, 1.f, 0.f);
            fwd.applyQuaternion(rifle->quaternion);
            rightV.applyQuaternion(rifle->quaternion).multiplyScalar(kCasingEjectSide);
            up.applyQuaternion(rifle->quaternion);
            auto cm = Mesh::create(casingGeo, casingMat);
            cm->castShadow = true;
            cm->position.copy(muzzle);                 // start near the barrel...
            cm->position.add(fwd * -0.33f).add(up * 0.03f).add(rightV * 0.05f);// ...pulled back to the breech, right side
            scene->add(cm);
            Casing c;
            c.mesh = cm;
            c.vel = rightV * frand(2.0f, 3.0f);        // flung out the side
            c.vel.add(up * frand(1.6f, 2.4f));         // and up
            c.vel.add(fwd * frand(-0.9f, -0.4f));      // and slightly back
            c.vel.add(Vector3(frand(-0.3f, 0.3f), 0.f, frand(-0.3f, 0.3f)));
            c.spinAxis.set(frand(-1.f, 1.f), frand(-1.f, 1.f), frand(-1.f, 1.f));
            if (c.spinAxis.length() < 1e-3f) c.spinAxis.set(0.f, 0.f, 1.f);
            c.spinAxis.normalize();
            c.spinRate = frand(14.f, 26.f);
            c.ttl = frand(1.8f, 2.4f);
            if (casings.size() >= kMaxCasings) {
                scene->remove(*casings.front().mesh);
                casings.erase(casings.begin());
            }
            casings.push_back(std::move(c));
        }

        if (!hasHit) return;

        PxRigidActor* actor = hitBuf.block.actor;
        Vector3 point = fromPxVec3(hitBuf.block.position);
        Vector3 hitNormal = fromPxVec3(hitBuf.block.normal);

        // enemy hit? -> blood spray
        if (actor && actor->userData) {
            auto* e = static_cast<Enemy*>(actor->userData);
            if (e->alive) {
                spawnParticles(point, dir, hitNormal, /*blood*/ true);
                e->hp--;
                hitMarkerT = 0.12f;
                hitWasKill = false;
                if (e->hp <= 0) {
                    killEnemy(e, dir);
                    score += 100;
                    hitMarkerT = 0.25f;// kill: longer, popped, red marker
                    hitWasKill = true;
                    scorePopT = 0.8f;  // float a "+100" up from the crosshair
                    sfx.thud.play(1.f, frand(0.95f, 1.05f));
                } else {
                    sfx.hit.play(frand(0.85f, 1.f), frand(0.92f, 1.08f));
                }
                return;
            }
        }

        // environment hit -> hard metal impact + dust + scorch decal
        sfx.metal.play(frand(0.75f, 1.f), frand(0.9f, 1.1f));
        spawnParticles(point, dir, hitNormal, /*blood*/ false);
        if (auto it = actorToMesh.find(actor); it != actorToMesh.end()) {
            stampDecal(it->second, point, hitNormal);
        }

        // knock the hit body around if it's dynamic
        if (auto* rd = actor ? actor->is<PxRigidDynamic>() : nullptr) {
            PxRigidBodyExt::addForceAtPos(*rd, toPxVec3(dir * 220.f),
                                          hitBuf.block.position, PxForceMode::eIMPULSE);
        }
    };

    // ===== grenade (G to throw): physics projectile + radial explosion =======
    auto grenadeGeo = SphereGeometry::create(0.09f, 12, 8);
    auto grenadeMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(0x2f3b24).roughness(0.55f).metalness(0.35f));
    auto blastGeo = SphereGeometry::create(0.6f, 12, 8);
    struct Grenade {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* body;
        float fuse;
    };
    std::vector<Grenade> grenades;
    float throwTimer = 0.f;      // >0 while a throw is in progress (also the cooldown)
    bool grenadeReleased = false;// has this throw spawned its projectile yet?
    float grenadeW = 0.f;        // smoothed weight of the additive throw overlay

    // launch a grenade body along the aim with an upward arc
    auto throwGrenade = [&]() {
        Vector3 pivot = playerPos + Vector3(-0.2f, kPlayerLen, 0);
        Vector3 dir = aimDir;// throw where the crosshair points
        auto m = Mesh::create(grenadeGeo, grenadeMat);
        m->position.copy(pivot + dir * 0.6f);
        m->castShadow = true;
        scene->add(m);
        auto* body = world.add(*m, 12.f);
        if (!body) {
            scene->remove(*m);
            return;
        }
        body->setLinearVelocity(toPxVec3(dir * kGrenadeSpeed + Vector3(0, 3.2f, 0)));
        body->setAngularVelocity(PxVec3(frand(-7.f, 7.f), frand(-7.f, 7.f), frand(-7.f, 7.f)));
        grenades.push_back({m, body, kGrenadeFuse});
    };

    // detonate at `at`: flash + dust + boom, kill enemies & knock props in radius
    auto explodeGrenade = [&](const Vector3& at) {
        sfx.boom.playAt(at, frand(0.94f, 1.06f));
        auto fm = MeshBasicMaterial::create();
        fm->color = Color(0xffd27a);
        fm->transparent = true;
        auto flash = Mesh::create(blastGeo, fm);
        flash->position.copy(at);
        flash->scale.set(2.2f, 2.2f, 2.2f);
        scene->add(flash);
        fx.push_back({flash, 0.12f});
        spawnParticles(at, Vector3(0, 1, 0), Vector3(0, 1, 0), /*blood*/ false);

        for (auto& e : enemies) {
            if (!e->alive) continue;
            Vector3 d = fromPxVec3(e->body->getGlobalPose().p) - at;
            if (d.length() < kBlastRadius) {
                d.y = 0.f;
                if (d.length() < 1e-3f) d.set(0, 0, 1);
                d.normalize();
                killEnemy(e.get(), d);
                score += 100;
                hitMarkerT = 0.25f;
                hitWasKill = true;
                scorePopT = 0.8f;
            }
        }
        for (auto& dyn : dynamics) {
            Vector3 d = fromPxVec3(dyn.body->getGlobalPose().p) - at;
            const float dist = d.length();
            if (dist < kBlastRadius && dist > 1e-3f) {
                d.normalize();
                dyn.body->addForce(toPxVec3(d * (1.f - dist / kBlastRadius) * 650.f), PxForceMode::eIMPULSE);
                dyn.body->wakeUp();
            }
        }
    };

    // ===== HUD (SVG overlay) ================================================
    auto ui = Scene::create();
    auto sz = canvas.size();
    auto uiCam = OrthographicCamera::create(0, sz.width(), sz.height(), 0, 0.1f, 100);
    uiCam->position.z = 10;
    Layout layout;
    FontLoader fontLoader;
    const Font font = fontLoader.defaultFont();

    // Scale a HUD widget group's baked SVG geometry + child offsets to physical
    // pixels, preserving any y-flip the group uses to map y-down SVG into the
    // y-up overlay. Screen-space sprites bypass the scene graph and are scaled
    // in makeText / explicitly instead.
    auto hudScale = [&](const std::shared_ptr<Object3D>& g) {
        g->scale.set(g->scale.x < 0.f ? -uiScale : uiScale,
                     g->scale.y < 0.f ? -uiScale : uiScale, 1.f);
    };

    // crosshair (4 ticks + dot) — DYNAMIC: the tick gap eases toward a spread
    // driven by recoil + movement (hip fire blooms, ADS tightens), and the
    // whole thing recolours on hit. Each tick lives in its own group so the
    // per-frame update just moves four groups.
    auto crosshair = Group::create();
    std::vector<std::shared_ptr<MeshBasicMaterial>> chMats;
    std::array<std::shared_ptr<Group>, 4> chTicks;
    {
        const float len = 9, th = 2;
        auto mkTick = [&](int i, float w, float h) {
            auto r = rect(w, h, 0xffffff, 0.9f);
            r.mesh->position.set(-w / 2, -h / 2, 0);// centre the rect on the group origin
            chTicks[i] = Group::create();
            chTicks[i]->add(r.mesh);
            crosshair->add(chTicks[i]);
            chMats.push_back(r.material);
        };
        mkTick(0, th, len);// up
        mkTick(1, th, len);// down
        mkTick(2, len, th);// left
        mkTick(3, len, th);// right
        auto dot = rect(3, 3, 0xffffff, 0.9f);
        dot.mesh->position.set(-1.5f, -1.5f, 0);
        crosshair->add(dot.mesh);
        chMats.push_back(dot.material);
    }
    ui->add(crosshair);
    layout.add(crosshair, 0.5f, 0.5f, 0, 0, 0.5f);

    // health bar (bottom-left): rounded bordered frame + damage-lag chip + fill.
    // The CHIP bar (amber) snaps to the old value and eases down toward the
    // fill, so every hit leaves a visible "chunk" that drains — classic
    // fighting-game damage feedback, and it makes burst damage readable.
    {
        auto fg = Group::create();
        fg->add(panel(224, 26, 6, kPanel, 0.75f, kPanelEdge, 1.5f));
        fg->scale.y = -1.f;
        hudScale(fg);
        ui->add(fg);
        layout.add(fg, 0.f, 0.f, 22, 56, 0.1f);
    }
    auto chipFill = rect(216, 18, 0xffaa55, 0.55f);
    {
        auto cg = Group::create();
        cg->add(chipFill.mesh);
        cg->scale.y = -1.f;
        hudScale(cg);
        ui->add(cg);
        layout.add(cg, 0.f, 0.f, 26, 52, 0.15f);
    }
    auto healthFill = rect(216, 18, kHudGood, 0.95f);
    {
        auto hg = Group::create();
        hg->add(healthFill.mesh);
        hg->scale.y = -1.f;
        hudScale(hg);
        ui->add(hg);
        layout.add(hg, 0.f, 0.f, 26, 52, 0.2f);
    }
    auto healthTxt = makeText(font, "100", 0xffffff, 15, 0.f, 0.f, 134, 43,
                              TextSprite::HorizontalAlignment::Center);
    ui->add(healthTxt);
    ui->add(makeText(font, "HP", kHudCyan, 12, 0.f, 0.f, 30, 70));

    // ammo (bottom-right)
    Readout ammoTxt{makeText(font, "", 0xffffff, 34, 1.f, 0.f, -40, 60,
                             TextSprite::HorizontalAlignment::Right)};
    ui->add(ammoTxt.sprite);
    ui->add(makeText(font, "AMMO", kHudCyan, 13, 1.f, 0.f, -40, 92,
                     TextSprite::HorizontalAlignment::Right));
    Readout reloadTxt{makeText(font, "", kHudWarn, 16, 1.f, 0.f, -40, 28,
                               TextSprite::HorizontalAlignment::Right)};
    ui->add(reloadTxt.sprite);

    // magazine pips (above the ammo readout): one per round, draining right to
    // left; a reload refills them as a left-to-right sweep timed to kReloadTime;
    // a dry mag pulses them red.
    auto magGroup = Group::create();
    std::vector<std::shared_ptr<MeshBasicMaterial>> pipMats;
    for (int i = 0; i < kMagSize; ++i) {
        auto p = rect(6, 20, 0xffffff, 0.95f);
        p.mesh->position.set(-static_cast<float>(i) * 10.f - 6.f, 0.f, 0.f);
        magGroup->add(p.mesh);
        pipMats.push_back(p.material);
    }
    hudScale(magGroup);
    ui->add(magGroup);
    layout.add(magGroup, 1.f, 0.f, -40, 112, 0.2f);

    // grenade pip (right of the health bar): drains on throw, refills over the
    // cooldown, green when ready
    auto gPip = rect(8, 26, kHudGood, 0.9f);
    {
        auto gg = Group::create();
        gg->add(gPip.mesh);
        hudScale(gg);
        ui->add(gg);
        layout.add(gg, 0.f, 0.f, 256, 30, 0.2f);
    }
    ui->add(makeText(font, "G", kHudCyan, 11, 0.f, 0.f, 258, 20));

    // score + enemies (top)
    Readout scoreTxt{makeText(font, "SCORE 0", kHudCyan, 22, 0.f, 1.f, 24, -34)};
    ui->add(scoreTxt.sprite);
    Readout aliveTxt{makeText(font, "", 0xffffff, 16, 1.f, 1.f, -24, -34,
                              TextSprite::HorizontalAlignment::Right)};
    ui->add(aliveTxt.sprite);

    // ── compass strip (top-centre) ──────────────────────────────────────────
    // Cardinal labels + 15° ticks slide horizontally with camera yaw under a
    // fixed caret; a numeric heading readout sits below the caret. Ticks are
    // SVG rects inside a top-centre-anchored group; labels are independent
    // screen-space TextSprites (they anchor themselves), both driven by the
    // same wrapped angular offset, hidden outside the strip's half-width.
    constexpr float kCompassHalfW = 170.f, kCompassPxPerRad = 150.f;
    struct CompassMark {
        std::shared_ptr<Mesh> tick;            // null for label-only marks
        std::shared_ptr<TextSprite> label;     // null for tick-only marks
        float ang = 0.f;                       // world bearing, radians
    };
    std::vector<CompassMark> compassMarks;
    auto compassTicks = Group::create();
    {
        static const char* kCard[8] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
        for (int d = 0; d < 360; d += 15) {
            CompassMark m;
            m.ang = static_cast<float>(d) * math::DEG2RAD;
            const bool card = d % 45 == 0;
            auto t = rect(2, card ? 12.f : 7.f, card ? kHudCyan : 0x9fb6c8, card ? 0.95f : 0.55f);
            m.tick = t.mesh;
            compassTicks->add(m.tick);
            if (card) {
                m.label = makeText(font, kCard[(d / 45) % 8], d == 0 ? kHudWarn : 0xd7e6f2,
                                   13, 0.5f, 1.f, 0, -46,
                                   TextSprite::HorizontalAlignment::Center);
                ui->add(m.label);
            }
            compassMarks.push_back(std::move(m));
        }
        // caret + baseline
        auto caret = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><polygon points="-5,0 5,0 0,-7" fill=")") +
                                   hex(kHudCyan) + R"("/></svg>)");
        compassTicks->add(caret);
        caret->position.set(0, -2, 0);
        auto base = rect(2.f * kCompassHalfW, 1.5f, kPanelEdge, 0.8f);
        base.mesh->position.set(-kCompassHalfW, -26, 0);
        compassTicks->add(base.mesh);
    }
    hudScale(compassTicks);
    ui->add(compassTicks);
    layout.add(compassTicks, 0.5f, 1.f, 0, -10, 0.2f);
    Readout headingTxt{makeText(font, "000", 0xd7e6f2, 13, 0.5f, 1.f, 0, -64,
                                TextSprite::HorizontalAlignment::Center)};
    ui->add(headingTxt.sprite);

    // ── radar (top-right) ───────────────────────────────────────────────────
    // Heading-up scope: rings + cross from one SVG document, a rotating sweep
    // wedge, hostile blips from a small pool, a north tick that orbits with
    // yaw, and the player as a centre dot. Same instrument the ocean demo
    // draws with ImGui — here it's pure SVG in the overlay pass.
    constexpr float kRadarR = 64.f, kRadarRange = 38.f;
    constexpr int kMaxBlips = 12;
    auto radar = Group::create();
    {
        std::ostringstream svg;
        svg << R"(<svg xmlns="http://www.w3.org/2000/svg">)"
            << R"(<circle cx="0" cy="0" r="64" fill=")" << hex(kPanel) << R"(" fill-opacity="0.55" stroke=")"
            << hex(kPanelEdge) << R"(" stroke-width="1.5"/>)"
            << R"(<circle cx="0" cy="0" r="42.7" fill="none" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1" stroke-opacity="0.7"/>)"
            << R"(<circle cx="0" cy="0" r="21.3" fill="none" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1" stroke-opacity="0.7"/>)"
            << R"(<line x1="-64" y1="0" x2="64" y2="0" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1" stroke-opacity="0.6"/>)"
            << R"(<line x1="0" y1="-64" x2="0" y2="64" stroke=")" << hex(kPanelEdge) << R"(" stroke-width="1" stroke-opacity="0.6"/>)"
            << R"(</svg>)";
        radar->add(svgFromString(svg.str()));
    }
    auto radarSweep = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><path d=")") +
                                    wedgePath(0.f, 62.f, 13.f) + R"(" fill=)" + '"' + hex(kHudCyan) +
                                    R"(" fill-opacity="0.16"/></svg>)");
    radar->add(radarSweep);
    auto radarNorth = rect(3, 8, kHudCyan, 0.9f);
    radarNorth.mesh->position.set(-1.5f, -4.f, 0.f);
    auto radarNorthG = Group::create();
    radarNorthG->add(radarNorth.mesh);
    radar->add(radarNorthG);
    {
        auto dotC = rect(5, 5, kHudCyan, 1.f);
        dotC.mesh->position.set(-2.5f, -2.5f, 0.f);
        radar->add(dotC.mesh);
    }
    std::vector<std::shared_ptr<Group>> blips;
    for (int i = 0; i < kMaxBlips; ++i) {
        auto b = rect(5, 5, kHudWarn, 0.95f);
        b.mesh->position.set(-2.5f, -2.5f, 0.f);// centre on the blip group origin
        auto bg = Group::create();
        bg->add(b.mesh);
        bg->visible = false;
        radar->add(bg);
        blips.push_back(bg);
    }
    hudScale(radar);
    ui->add(radar);
    layout.add(radar, 1.f, 1.f, -92, -110, 0.15f);

    // hit marker (centre, flashes)
    auto hitMarker = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><g stroke=")") + hex(kHudWarn) +
                                   R"(" stroke-width="3"><line x1="-11" y1="-11" x2="-4" y2="-4"/><line x1="11" y1="-11" x2="4" y2="-4"/>)" +
                                   R"(<line x1="-11" y1="11" x2="-4" y2="4"/><line x1="11" y1="11" x2="4" y2="4"/></g></svg>)");
    hitMarker->visible = false;
    ui->add(hitMarker);
    layout.add(hitMarker, 0.5f, 0.5f, 0, 0, 0.6f);
    auto hitMats = svgMats(hitMarker);// recolour: white = hit, red + pop = kill

    // "+100" kill pop — floats up from beside the crosshair and vanishes
    auto scorePop = makeText(font, "+100", kHudGood, 16, 0.5f, 0.5f, 30, 14,
                             TextSprite::HorizontalAlignment::Left);
    scorePop->visible = false;
    ui->add(scorePop);

    // damage-direction arcs (pool built from the wedge primitive); each is
    // rotated toward its attacker bearing relative to the camera and faded
    for (auto& a : dmgArcs) {
        a.g = svgFromString(std::string(R"(<svg xmlns="http://www.w3.org/2000/svg"><path d=")") +
                            wedgePath(56.f, 68.f, 26.f) + R"(" fill=)" + '"' + hex(kHudWarn) + R"("/></svg>)");
        a.mats = svgMats(a.g);
        a.g->visible = false;
        hudScale(a.g);
        ui->add(a.g);
        layout.add(a.g, 0.5f, 0.5f, 0, 0, 0.55f);
    }

    // low-health vignette (full-screen red, opacity tracks damage)
    auto vignette = rect(1, 1, kHudWarn, 0.f);
    vignette.mesh->position.set(0, 0, 0);
    ui->add(vignette.mesh);
    layout.addRaw([m = vignette.mesh](float W, float H) { m->scale.set(W, H, 1); m->position.set(0, 0, 0.05f); });

    // controls hint (bottom-centre)
    ui->add(makeText(font, "WASD move    MOUSE look    LMB fire    SHIFT sprint    SPACE jump    R reload    G grenade",
                     0x9fb6c8, 12, 0.5f, 0.f, 0, 20,
                     TextSprite::HorizontalAlignment::Center));

    // game-over: full-screen dim backdrop (eases in) + bordered rounded panel
    auto overDim = rect(1, 1, 0x000000, 0.f);
    ui->add(overDim.mesh);
    layout.addRaw([m = overDim.mesh](float W, float H) { m->scale.set(W, H, 1); m->position.set(0, 0, 0.65f); });
    auto over = Group::create();
    over->visible = false;
    ui->add(over);
    {
        auto pg = Group::create();
        pg->add(panel(420, 210, 10, kPanel, 0.92f, kPanelEdge, 2.f));
        pg->scale.y = -1.f;
        hudScale(pg);
        over->add(pg);
        layout.add(pg, 0.5f, 0.5f, -210, 105, 0.7f);
        auto sep = rect(340, 1.5f, kPanelEdge, 0.9f);
        auto sg = Group::create();
        sg->add(sep.mesh);
        hudScale(sg);
        over->add(sg);
        layout.add(sg, 0.5f, 0.5f, -170, 30, 0.71f);
    }
    auto overTitle = makeText(font, "GAME OVER", kHudWarn, 40, 0.5f, 0.5f, 0, 50,
                              TextSprite::HorizontalAlignment::Center);
    over->add(overTitle);
    Readout overScore{makeText(font, "", 0xffffff, 22, 0.5f, 0.5f, 0, 0,
                               TextSprite::HorizontalAlignment::Center)};
    over->add(overScore.sprite);
    over->add(makeText(font, "PRESS ENTER OR CLICK RESTART", kHudCyan, 15, 0.5f, 0.5f, 0, -40,
                       TextSprite::HorizontalAlignment::Center));
    // restart hit-target (SpriteInteractor)
    auto restartLabel = makeText(font, "[ RESTART ]", kHudGood, 20, 0.5f, 0.5f, 0, -75,
                                 TextSprite::HorizontalAlignment::Center);
    over->add(restartLabel);
    auto restartHitMat = SpriteMaterial::create();
    restartHitMat->visible = false;
    auto restartHit = Sprite::create(restartHitMat);
    restartHit->screenSpace = true;
    restartHit->screenAnchor.set(0.5f, 0.5f);
    restartHit->scale.set(180 * uiScale, 44 * uiScale, 1);// screen-space: bypasses parent scale
    restartHit->center.set(0.5f, 0.5f);
    over->add(restartHit);

    SpriteInteractor interactor(canvas, *ui);

    // ===== restart ==========================================================
    std::function<void()> restart = [&]() {
        health = 100;
        ammo = kMagSize;
        score = 0;
        reloading = false;
        gameOver = false;
        over->visible = false;
        // clear enemies
        for (auto& e : enemies) removeEnemy(e.get());
        enemies.clear();
        // clear in-flight grenades
        for (auto& g : grenades) {
            world.unbind(*g.mesh);
            world.scene().removeActor(*g.body);
            g.body->release();
            scene->remove(*g.mesh);
        }
        grenades.clear();
        for (auto& c : casings) scene->remove(*c.mesh);
        casings.clear();
        throwTimer = 0.f;
        grenadeReleased = false;
        recoilKick = 0.f;
        recoilYaw = 0.f;
        aiming = false;
        inspect = false;
        chipHealth = 100.f;
        scorePopT = 0.f;
        hitWasKill = false;
        for (auto& a : dmgArcs) a.t = 0.f;
        // reset player
        playerBody->setGlobalPose(toPxTransform(Vector3(0, kPlayerHalf, 0)));
        playerBody->setLinearVelocity(PxVec3(0));
        // reset dynamic props
        for (auto& d : dynamics) {
            d.body->setGlobalPose(toPxTransform(d.home));
            d.body->setLinearVelocity(PxVec3(0));
            d.body->setAngularVelocity(PxVec3(0));
            d.body->wakeUp();
        }
        setCursorLocked(true);// re-grab the mouse
    };

    restartHit->onMouseUp = [&](int) {
        if (gameOver) restart();
    };

    // ===== input ============================================================
    // mouse look via frame-to-frame delta (pointer-locked: raw virtual deltas)
    MouseMoveListener look([&](const Vector2& p) {
        if (gameOver) {// don't spin the camera while aiming for the restart button
            lastMouse = p;
            haveMouse = true;
            return;
        }
        if (haveMouse) {
            const float dx = p.x - lastMouse.x;
            const float dy = p.y - lastMouse.y;
            const float sens = kMouseSens * (1.f - 0.45f * zoomT);// finer aim while zoomed
            camYaw -= dx * sens;
            camPitch += dy * sens;
            camPitch = std::clamp(camPitch, -0.2f, 1.2f);
        }
        lastMouse = p;
        haveMouse = true;
    });
    canvas.addMouseListener(look);

    MouseDownListener down([&](int button, const Vector2&) {
        if (button == 0) firing = true;
        else if (button == 1) aiming = true;     // RMB -> aim-down-sights
        else if (button == 2) inspect = true;    // middle (wheel press) -> face the player
    });
    MouseUpListener up([&](int button, const Vector2&) {
        if (button == 0) {
            firing = false;
            firedEmpty = false;
        } else if (button == 1) {
            aiming = false;
        } else if (button == 2) {
            inspect = false;
        }
    });
    canvas.addMouseListener(down);
    canvas.addMouseListener(up);

    struct WheelListener: MouseListener {
        std::function<void(float)> f;
        void onMouseWheel(const Vector2& d) override { f(d.y); }
    } wheel;
    wheel.f = [&](float dy) { camDist = std::clamp(camDist - dy * 0.6f, 3.f, 11.f); };
    canvas.addMouseListener(wheel);

    bool jumpQueued = false;
    canvas.onKeyPressed([&](KeyEvent e) {
        if (e.key == Key::R && !reloading && ammo < kMagSize) {
            reloading = true;
            reloadTimer = kReloadTime;
            sfx.reload.play();
        } else if (e.key == Key::SPACE && !gameOver) {
            jumpQueued = true;
        } else if (e.key == Key::G && !gameOver && throwTimer <= 0.f) {
            throwTimer = kThrowTime;
            grenadeReleased = false;
            if (pa.grenade) pa.grenade->reset();// restart the throw overlay from frame 0
        } else if (e.key == Key::ENTER && gameOver) {
            restart();
        }
    });

    // ===== resize ===========================================================
    auto relayout = [&](WindowSize s) {
        const float W = static_cast<float>(s.width());
        const float H = static_cast<float>(s.height());
        camera->aspect = s.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(s);
        uiCam->right = W;
        uiCam->top = H;
        uiCam->updateProjectionMatrix();
        layout.apply(W, H);
    };
    canvas.onWindowResize([&](WindowSize s) { relayout(s); });
    relayout(sz);

    setCursorLocked(true);// grab the mouse for FPS-style look

    // ===== main loop ========================================================
    Clock clock;
    canvas.animate([&] {
        float dt = clock.getDelta();
        if (dt > 0.05f) dt = 0.05f;// clamp big hitches

        // --- read player body pose ---
        const PxTransform pt = playerBody->getGlobalPose();
        playerPos.set(pt.p.x, pt.p.y, pt.p.z);

        // ground check (shared by jump + jump animation)
        PxRaycastBuffer gb;
        const bool grounded = world.scene().raycast(pt.p, PxVec3(0, -1, 0), kPlayerHalf + 0.18f, gb) && gb.hasBlock;

        // --- movement input (camera-relative; ground only — no air control) ---
        float moveSpeed = 0.f;
        inFwd = 0.f;
        inStr = 0.f;
        if (!gameOver) {
            PxVec3 vel = playerBody->getLinearVelocity();
            if (grounded) {
                const Vector3 F(std::sin(camYaw), 0, std::cos(camYaw));
                const Vector3 R(-std::cos(camYaw), 0, std::sin(camYaw));// screen-right = cross(F, up)
                float fwd = (canvas.isKeyDown(Key::W) ? 1.f : 0.f) - (canvas.isKeyDown(Key::S) ? 1.f : 0.f);
                float str = (canvas.isKeyDown(Key::D) ? 1.f : 0.f) - (canvas.isKeyDown(Key::A) ? 1.f : 0.f);
                inFwd = fwd;
                inStr = str;
                Vector3 move = F * fwd + R * str;
                const bool sprint = canvas.isKeyDown(Key::LEFT_SHIFT);
                const float spd = sprint ? kRunSpeed : kWalkSpeed;
                if (move.length() > 0.01f) {
                    move.normalize();
                    vel.x = move.x * spd;
                    vel.z = move.z * spd;
                    moveSpeed = spd;
                } else {
                    vel.x = 0;
                    vel.z = 0;
                }
                if (jumpQueued) vel.y = kJumpSpeed;
            }
            // Airborne: leave horizontal velocity alone so takeoff momentum
            // carries, but the player can't steer or accelerate mid-jump.
            playerBody->setLinearVelocity(vel);
        }
        jumpQueued = false;

        // --- firing ---
        fireTimer -= dt;
        hitReactTimer = std::max(0.f, hitReactTimer - dt);

        // grenade throw: advance the timer; release the projectile partway through
        if (throwTimer > 0.f) {
            throwTimer = std::max(0.f, throwTimer - dt);
            if (!grenadeReleased && throwTimer <= kThrowTime - kThrowRelease) {
                throwGrenade();
                grenadeReleased = true;
            }
        }
        if (reloading) {
            reloadTimer -= dt;
            if (reloadTimer <= 0.f) {
                reloading = false;
                ammo = kMagSize;
            }
        }
        if (firing && !gameOver) {
            if (ammo > 0 && !reloading && fireTimer <= 0.f) {
                fire();
            } else if (ammo == 0 && !reloading && !firedEmpty) {
                sfx.empty.play();
                firedEmpty = true;
                reloading = true;// auto-reload
                reloadTimer = kReloadTime;
                sfx.reload.play();
            }
        }

        // --- step physics ---
        world.step(dt);

        // re-read after step for visuals
        const PxTransform pt2 = playerBody->getGlobalPose();
        playerPos.set(pt2.p.x, pt2.p.y, pt2.p.z);

        // --- place + animate the SWAT player ---
        playerRig->position.set(playerPos.x, playerPos.y - kPlayerHalf, playerPos.z);
        playerRig->rotation.y = camYaw + kModelYaw;
        if (mixer) {
            // BASE layer (full body): directional locomotion / jump / hit-flinch.
            // Reload + fire are additive OVERLAYS handled below, so the legs keep
            // moving while the arms reload/fire.
            AnimationAction* want = pa.idle;
            if (moveSpeed > 0.1f) {
                const bool sprint = moveSpeed > kWalkSpeed + 0.5f;
                if (std::abs(inStr) > std::abs(inFwd) + 0.01f) {// sideways dominant
                    want = inStr > 0.f ? (pa.strafeR ? pa.strafeR : pa.walk)
                                       : (pa.strafeL ? pa.strafeL : pa.walk);
                } else if (inFwd < 0.f) {// backpedal
                    want = (sprint && pa.runBack) ? pa.runBack : (pa.walkBack ? pa.walkBack : pa.walk);
                } else {// forward
                    want = (sprint && pa.run) ? pa.run : (pa.walk ? pa.walk : pa.idle);
                }
            }
            if (!grounded && pa.jump) want = pa.jump;
            if (hitReactTimer > 0.f && pa.hit) want = pa.hit;

            if (want && want != currentA) {
                want->reset();
                want->play();
                if (currentA) currentA->crossFadeTo(want, 0.18f);
                currentA = want;
            }

            // ADDITIVE OVERLAYS: reload + fire blend on top of the base layer (legs
            // walk, arms reload/fire). Restart the reload from frame 0 each time it
            // begins; ease the weights so they fade in/out instead of popping.
            if (reloading && !wasReloading && pa.reload) pa.reload->reset();
            wasReloading = reloading;
            const float reloadTarget = reloading ? 1.f : 0.f;
            const float fireTarget = (firing && !reloading && throwTimer <= 0.f && ammo > 0 && pa.fire) ? 1.f : 0.f;
            const float grenadeTarget = (throwTimer > 0.f && pa.grenade) ? 1.f : 0.f;
            reloadW += (reloadTarget - reloadW) * std::min(1.f, dt * 12.f);
            fireW += (fireTarget - fireW) * std::min(1.f, dt * 20.f);
            grenadeW += (grenadeTarget - grenadeW) * std::min(1.f, dt * 14.f);
            if (pa.reload) pa.reload->setEffectiveWeight(reloadW);
            if (pa.fire) pa.fire->setEffectiveWeight(fireW);
            if (pa.grenade) pa.grenade->setEffectiveWeight(grenadeW);
            mixer->update(dt);

            // Neutralise clip root motion: the locomotion clips translate the Hips,
            // but physics owns the player position — so the model would drift then
            // snap back on each loop. The skeleton is Z-up rotated to Y-up by the
            // static Armature node, so Hips LOCAL X = world-lateral and LOCAL Y =
            // world-forward; pin both to bind to kill horizontal drift. Keep LOCAL Z
            // (= world-up) so the vertical bob survives.
            if (hipsBone) {
                hipsBone->position.x = hipsBind.x;
                hipsBone->position.y = hipsBind.y;
            }

            // Aim tilt: pitch the upper spine by the aim angle so the held rifle
            // (and both hands gripping it) track the target vertically — body yaw
            // already handles the horizontal. Applied as a world-space rotation
            // about the player's lateral axis, converted into the spine's local
            // frame, so it's robust to the imported bone orientation.
            if (spineBone && spineBone->parent) {
                playerRig->updateMatrixWorld(true);// make the parent's world current post-pose
                const Vector3 lateral(-std::cos(camYaw), 0.f, std::sin(camYaw));// player's right
                const float aimPitch = std::asin(std::clamp(aimDir.y, -1.f, 1.f));
                Quaternion rWorld;
                rWorld.setFromAxisAngle(lateral, aimPitch * kSpinePitchGain);
                Vector3 pPos, pScl;
                Quaternion pQuat;
                spineBone->parent->matrixWorld->decompose(pPos, pQuat, pScl);
                Quaternion rLocal = pQuat.clone().invert();
                rLocal.multiply(rWorld).multiply(pQuat);
                spineBone->quaternion.premultiply(rLocal);
            }
        }
        const bool justLanded = grounded && !wasGrounded;
        wasGrounded = grounded;

        // Pin the rifle into the right hand (grip) and aim the barrel (+Z) straight
        // down the line to the LEFT hand — so the support hand lands on the barrel /
        // fore-grip and the stock sits behind the grip, at the shoulder. Both hands
        // were just tilted by the spine aim, so the whole rig tracks the target.
        // Falls back to aiming along aimDir if the left-hand bone wasn't found.
        if (handBone && rifle->visible) {
            playerRig->updateMatrixWorld(true);// propagate the spine aim tilt to the hands
            Vector3 hp, hs;
            Quaternion hq;
            handBone->matrixWorld->decompose(hp, hq, hs);
            Vector3 off = kRifleGripPos;
            off.applyQuaternion(hq);// small grip nudge, in hand space
            rifle->position.copy(hp).add(off);
            // During a reload or grenade throw the left hand leaves the barrel (mag
            // swap / wind-up), so following it whips the gun around — keep the gun
            // pointing forward along aimDir until the support hand is back on the rifle.
            const bool twoHand = leftHandBone && !reloading && throwTimer <= 0.f;
            if (twoHand) {
                Vector3 lp, ls;
                Quaternion lq;
                leftHandBone->matrixWorld->decompose(lp, lq, ls);
                rifle->lookAt(lp);// barrel runs grip -> fore-grip (through the left hand)
            } else {
                Vector3 aimTarget = rifle->position;
                aimTarget.add(aimDir);
                rifle->lookAt(aimTarget);
            }
            // Extra spin about the rifle's own up axis (model-orientation fix). The
            // forward (aimDir) reload/throw pose needs a different cant than the
            // two-handed hold, so pick per case.
            const float upRot = twoHand ? kRifleUpRotDeg : kRifleUpRotReloadDeg;
            rifle->quaternion.multiply(
                    Quaternion().setFromAxisAngle(Vector3(0.f, 1.f, 0.f), math::degToRad(upRot)));
        }

        // footstep sfx: voices cycle the synth variants; on top, alternate feet
        // get a small fixed pitch split (the classic L/R trick) plus a per-step
        // jitter, and sprinting steps land harder than walking ones.
        if (!gameOver && justLanded) {
            sfx.step.play(1.2f, frand(0.78f, 0.85f));// landing: one heavy, low step
            stepTimer = 0.25f;
        }
        if (!gameOver && moveSpeed > 0.1f) {
            stepTimer -= dt;
            if (stepTimer <= 0.f) {
                const bool sprint = moveSpeed > kWalkSpeed + 0.5f;
                stepLeft = !stepLeft;
                sfx.step.play(sprint ? 1.f : 0.65f,
                              (stepLeft ? 0.96f : 1.04f) * frand(0.94f, 1.06f));
                stepTimer = sprint ? 0.33f : 0.62f;
            }
        } else {
            stepTimer = 0.12f;// idle: first step lands right after movement starts
        }

        // --- camera rig (over-the-shoulder third person; RMB = aim-down-sights) ---
        // recoil eases back to zero; while it's non-zero it raises the aim (lower
        // effective pitch tilts the third-person camera/look up) and jitters the yaw.
        {
            const float r = std::min(1.f, dt * kRecoilRecover);
            recoilKick -= recoilKick * r;
            recoilYaw -= recoilYaw * r;
        }
        // ADS zoom eases in/out: pull the camera in, narrow the FOV, tighten the shoulder.
        zoomT += (((aiming && !gameOver) ? 1.f : 0.f) - zoomT) * std::min(1.f, dt * kZoomSpeed);
        // middle-mouse "inspect": swing the camera around toward the player's front.
        inspectT += (((inspect && !gameOver) ? 1.f : 0.f) - inspectT) * std::min(1.f, dt * kInspectSpeed);
        const float effDist = camDist + (kCamDistAds - camDist) * zoomT;
        const float shoulder = kCamShoulder + (kCamShoulderAds - kCamShoulder) * zoomT;
        camera->fov = kFovHip + (kFovAds - kFovHip) * zoomT;
        camera->updateProjectionMatrix();
        crosshair->scale.set(uiScale * (1.f + recoilKick * 4.f), uiScale * (1.f + recoilKick * 4.f), 1.f);// bloom on recoil
        const float effYaw = camYaw + recoilYaw;
        const float effPitch = camPitch - recoilKick;
        const Vector3 pivot = playerPos + Vector3(0, kPlayerLen, 0);
        // Aim ray always comes from the normal (un-swung) view, so shooting still goes
        // forward even while the inspect camera is facing the player.
        const Vector3 aimOff(
                -std::sin(effYaw) * std::cos(effPitch) * effDist,
                std::sin(effPitch) * effDist + 0.4f,
                -std::cos(effYaw) * std::cos(effPitch) * effDist);
        aimDir.copy(aimOff).multiplyScalar(-1.f).normalize();// camera forward = the crosshair ray
        // Camera placement: yaw swings up to 180° toward the player while inspecting,
        // so the camera ends up in front looking back at the character's face. The
        // screen-right shoulder slide keeps the over-the-shoulder framing otherwise.
        const float viewYaw = effYaw + inspectT * math::PI;
        const Vector3 camOff(
                -std::sin(viewYaw) * std::cos(effPitch) * effDist,
                std::sin(effPitch) * effDist + 0.4f,
                -std::cos(viewYaw) * std::cos(effPitch) * effDist);
        const Vector3 shoulderOff = Vector3(-std::cos(viewYaw), 0.f, std::sin(viewYaw)) * shoulder;
        // Camera-wall collision: cast from the player out to the desired camera spot;
        // if static level geometry blocks the boom, pull the camera in to just before
        // it. Snap in instantly (never clip through), ease back out once it clears.
        const Vector3 boom = camOff + shoulderOff;// pivot -> desired camera
        const float wishDist = boom.length();
        if (wishDist > 1e-4f) {
            Vector3 boomDir = boom;
            boomDir.multiplyScalar(1.f / wishDist);
            float target = wishDist;
            PxRaycastBuffer hb;
            PxQueryFilterData fd;
            fd.flags = PxQueryFlags(PxQueryFlag::eSTATIC);// level geometry only (no player/enemies/crates)
            if (world.scene().raycast(toPxVec3(pivot), toPxVec3(boomDir), wishDist, hb,
                                      PxHitFlags(PxHitFlag::eDEFAULT), fd) &&
                hb.hasBlock) {
                target = std::clamp(hb.block.distance - kCamSkin, kCamMinDist, wishDist);
            }
            if (target < camBoom) camBoom = target;// snap in immediately so we never clip
            else camBoom += (target - camBoom) * std::min(1.f, dt * kCamReturnSpeed);// ease back out
            camera->position.copy(pivot).add(boomDir * camBoom);
        } else {
            camera->position.copy(pivot + camOff + shoulderOff);
        }
        camera->lookAt(pivot + shoulderOff);

        // --- enemies ---
        if (!gameOver) {
            enemySpawnTimer -= dt;
            int aliveCount = 0;
            for (auto& e : enemies)
                if (e->alive) aliveCount++;
            if (aliveCount < kMaxEnemies && enemySpawnTimer <= 0.f) {
                spawnEnemy();
                enemySpawnTimer = 1.4f;
            }
        }
        // refresh the flow field whenever the player crosses into a new cell
        {
            const int pr = colOf(playerPos.z), pc = colOf(playerPos.x);
            if (pr != navPlayerRow || pc != navPlayerCol) {
                navPlayerRow = pr;
                navPlayerCol = pc;
                rebuildFlow(pr, pc);
            }
        }
        for (auto& e : enemies) {
            const PxVec3 ep = e->body->getGlobalPose().p;
            if (e->alive) {
                Vector3 toPlayer(playerPos.x - ep.x, 0, playerPos.z - ep.z);
                const float d = toPlayer.length();
                PxVec3 v = e->body->getLinearVelocity();
                if (d > kEnemyAttackRange) {
                    Vector3 desired(toPlayer.x, 0.f, toPlayer.z);
                    desired.normalize();
                    // Steer down the flow-field gradient (toward lower step-count =
                    // toward the player, routed around static cover). distAt() treats
                    // blocked / unreachable cells as very far, so the gradient also
                    // points away from walls. Falls back to straight-in if our own
                    // cell is unreachable (e.g. shoved into geometry).
                    const int er = colOf(ep.z), ec = colOf(ep.x);
                    auto rawDist = [&](int r, int c) -> float {
                        if (r < 0 || r >= gridN || c < 0 || c >= gridN) return 1e6f;
                        const int dv = navDist[navIdx(r, c)];
                        return dv < 0 ? 1e6f : static_cast<float>(dv);
                    };
                    Vector3 dir = desired;
                    const float here = rawDist(er, ec);
                    if (here < 1e6f) {
                        // clamp blocked / unreachable neighbours to a few steps worse than
                        // "here" so walls repel smoothly instead of snapping us to an axis
                        auto g = [&](int r, int c) { return std::min(rawDist(r, c), here + 3.f); };
                        Vector3 flow(g(er, ec - 1) - g(er, ec + 1), 0.f,
                                     g(er - 1, ec) - g(er + 1, ec));
                        if (flow.length() > 1e-3f) {
                            flow.normalize();
                            dir = flow;
                        }
                    }
                    // separation: ease apart from other live bots so they don't stack
                    // up while funnelling along the same path
                    for (auto& o : enemies) {
                        if (o.get() == e.get() || !o->alive) continue;
                        const PxVec3 op = o->body->getGlobalPose().p;
                        Vector3 away(ep.x - op.x, 0.f, ep.z - op.z);
                        const float sd = away.length();
                        if (sd > 1e-3f && sd < kSeparation)
                            dir += away * ((kSeparation - sd) / sd * 0.6f);
                    }
                    if (dir.length() < 1e-3f) dir = desired;
                    dir.normalize();
                    v.x = dir.x * kEnemySpeed;
                    v.z = dir.z * kEnemySpeed;
                } else {
                    v.x = 0;
                    v.z = 0;
                    e->attackCd -= dt;
                    if (e->attackCd <= 0.f && !gameOver) {
                        health -= 9;
                        hitReactTimer = 0.45f;// flinch
                        e->attackCd = 0.8f;
                        // Damage-direction arc: record the attacker's bearing;
                        // the HUD rotates a wedge toward them and fades it.
                        {
                            const PxVec3 ap = e->body->getGlobalPose().p;
                            auto& slot = dmgArcs[dmgArcNext];
                            dmgArcNext = static_cast<int>((dmgArcNext + 1) % dmgArcs.size());
                            slot.t = 1.f;
                            slot.bearing = std::atan2(ap.x - playerPos.x, ap.z - playerPos.z);
                        }
                        sfx.hurt.play(1.f, frand(0.95f, 1.05f));
                        if (health <= 0) {
                            health = 0;
                            gameOver = true;
                            over->visible = true;
                            overScore.set("FINAL SCORE   " + std::to_string(score));
                            setCursorLocked(false);// release the mouse for the restart UI
                        }
                    }
                }
                e->body->setLinearVelocity(v);
            } else {
                e->deadTtl -= dt;
            }
        }
        // remove expired dead enemies
        for (auto it = enemies.begin(); it != enemies.end();) {
            if (!(*it)->alive && (*it)->deadTtl <= 0.f) {
                removeEnemy(it->get());
                it = enemies.erase(it);
            } else {
                ++it;
            }
        }

        // --- grenades: tick fuses, detonate, recycle ---
        for (auto it = grenades.begin(); it != grenades.end();) {
            it->fuse -= dt;
            if (it->fuse <= 0.f) {
                explodeGrenade(fromPxVec3(it->body->getGlobalPose().p));
                world.unbind(*it->mesh);
                world.scene().removeActor(*it->body);
                it->body->release();
                scene->remove(*it->mesh);
                it = grenades.erase(it);
            } else {
                ++it;
            }
        }

        // --- transient fx ---
        for (auto it = fx.begin(); it != fx.end();) {
            it->ttl -= dt;
            if (it->ttl <= 0.f) {
                scene->remove(*it->obj);
                it = fx.erase(it);
            } else {
                ++it;
            }
        }

        // --- ejected casings (ballistic + tumble; settle on the ground) ---
        for (auto it = casings.begin(); it != casings.end();) {
            auto& c = *it;
            c.ttl -= dt;
            if (c.ttl <= 0.f) {
                scene->remove(*c.mesh);
                it = casings.erase(it);
                continue;
            }
            c.vel.y -= 9.81f * dt;
            c.mesh->position.add(c.vel * dt);
            if (c.mesh->position.y < 0.02f && c.vel.y < 0.f) {
                c.mesh->position.y = 0.02f;
                c.vel.y = -c.vel.y * 0.35f;// damped bounce
                c.vel.x *= 0.6f;
                c.vel.z *= 0.6f;
                c.spinRate *= 0.5f;
            }
            Quaternion dq;
            dq.setFromAxisAngle(c.spinAxis, c.spinRate * dt);
            c.mesh->quaternion.premultiply(dq);
            ++it;
        }

        // --- impact particles (dust / blood) ---
        for (auto it = bursts.begin(); it != bursts.end();) {
            auto& b = *it;
            b.ttl -= dt;
            if (b.ttl <= 0.f) {
                scene->remove(*b.group);
                it = bursts.erase(it);
                continue;
            }
            for (size_t i = 0; i < b.pos.size(); ++i) {
                b.vel[i].y -= b.gravity * dt;
                b.vel[i] *= std::max(0.f, 1.f - b.drag * dt);
                b.pos[i] += b.vel[i] * dt;
                b.sprites[i]->position.copy(b.pos[i]);
            }
            b.mat->opacity = std::clamp(b.ttl / b.life, 0.f, 1.f);
            ++it;
        }

        // --- HUD update ---
        hudT += dt;

        // health: fill + colour + low-HP pulse; chip bar eases down toward it
        healthFill.mesh->scale.x = std::max(0.001f, static_cast<float>(health) / 100.f);
        healthFill.material->color = Color(health > 50 ? kHudGood : (health > 25 ? 0xffaa33 : kHudWarn));
        healthFill.material->opacity = health <= 25 ? 0.7f + 0.25f * std::sin(hudT * 8.f) : 0.95f;
        if (static_cast<float>(health) > chipHealth) chipHealth = static_cast<float>(health);
        chipHealth = std::max(static_cast<float>(health), chipHealth - dt * 35.f);
        chipFill.mesh->scale.x = std::max(0.001f, chipHealth / 100.f);
        healthTxt->setText(std::to_string(health));

        // ammo readout + magazine pips (reload = left-to-right refill sweep)
        {
            std::ostringstream os;
            os << ammo << " / " << kMagSize;
            ammoTxt.set(os.str());
        }
        reloadTxt.set(reloading ? "RELOADING" : (ammo == 0 ? "EMPTY" : ""));
        {
            const int lit = reloading
                ? static_cast<int>((1.f - reloadTimer / kReloadTime) * kMagSize + 1e-3f)
                : ammo;
            const bool dry = !reloading && ammo == 0;
            for (int i = 0; i < kMagSize; ++i) {
                const bool on = i < lit;
                pipMats[i]->color = Color(dry ? kHudWarn : (on ? 0xffffff : kPanelEdge));
                pipMats[i]->opacity = dry ? 0.25f + 0.4f * (0.5f + 0.5f * std::sin(hudT * 10.f))
                                          : (on ? 0.95f : 0.35f);
            }
        }

        // grenade pip: drains on throw, refills over the cooldown
        {
            const float ready = throwTimer <= 0.f ? 1.f : 1.f - throwTimer / kThrowTime;
            gPip.mesh->scale.y = std::max(0.05f, ready);
            gPip.material->color = Color(ready >= 1.f ? kHudGood : 0x9fb6c8);
        }

        scoreTxt.set("SCORE " + std::to_string(score));
        {
            int a = 0;
            for (auto& e : enemies)
                if (e->alive) a++;
            aliveTxt.set("HOSTILES " + std::to_string(a));
        }

        // crosshair: spread eases toward recoil + movement (ADS tightens)
        {
            const PxVec3 pv = playerBody->getLinearVelocity();
            const float speed = std::sqrt(pv.x * pv.x + pv.z * pv.z);
            const float target = std::clamp((aiming ? 2.5f : 6.f) + recoilKick * 60.f + speed * 0.9f, 2.f, 26.f);
            chSpread += (target - chSpread) * std::min(1.f, dt * 14.f);
            const float s = chSpread + 4.5f;
            chTicks[0]->position.set(0, s, 0);
            chTicks[1]->position.set(0, -s, 0);
            chTicks[2]->position.set(-s, 0, 0);
            chTicks[3]->position.set(s, 0, 0);
        }

        // hit marker: white flash on hit, red popped flash on kill; "+100" floats
        hitMarkerT -= dt;
        hitMarker->visible = hitMarkerT > 0.f;
        if (hitMarker->visible) {
            const float pop = hitWasKill ? 1.f + 0.6f * std::min(1.f, hitMarkerT / 0.25f) : 1.f;
            hitMarker->scale.set(uiScale * pop, uiScale * pop, 1);
            for (auto& m : hitMats) m->color = Color(hitWasKill ? kHudWarn : 0xffffff);
        }
        scorePopT -= dt;
        scorePop->visible = scorePopT > 0.f;
        if (scorePop->visible) scorePop->position.set(30.f * uiScale, (14.f + (0.8f - scorePopT) * 55.f) * uiScale, 0.f);// screen-space sprite

        // damage-direction arcs: rotate toward the attacker (camera-relative), fade
        for (auto& a : dmgArcs) {
            a.t -= dt;
            a.g->visible = a.t > 0.f;
            if (!a.g->visible) continue;
            a.g->rotation.z = wrapPi(a.bearing - camYaw);
            const float o = std::clamp(a.t, 0.f, 1.f) * 0.8f;
            for (auto& m : a.mats) m->opacity = o;
        }

        // compass: slide marks under the caret; heading readout
        {
            for (auto& m : compassMarks) {
                // screen-right = bearing-(90°) in this scene's mirrored frame,
                // so marks move LEFT as bearing increases (matches the radar)
                const float dx = -wrapPi(m.ang - camYaw) * kCompassPxPerRad;
                const bool vis = std::abs(dx) < kCompassHalfW;
                if (m.tick) {
                    m.tick->visible = vis;
                    m.tick->position.set(dx - 1.f, m.label ? -16.f : -12.f, 0.f);
                }
                if (m.label) {
                    m.label->visible = vis;
                    // label is a screen-space sprite (bypasses the compassTicks
                    // group scale that the tick rides), so scale dx + offset here
                    m.label->position.set(dx * uiScale, -46.f * uiScale, 0.f);
                }
            }
            const int deg = (static_cast<int>(std::round(camYaw * math::RAD2DEG)) % 360 + 360) % 360;
            std::ostringstream os;
            os << std::setw(3) << std::setfill('0') << deg;
            headingTxt.set(os.str());
        }

        // radar: sweep spins, north tick orbits, hostiles blip (heading-up)
        {
            radarSweep->rotation.z = -hudT * 2.2f;
            const float nRel = wrapPi(0.f - camYaw);
            radarNorthG->position.set(-std::sin(nRel) * (kRadarR - 7.f), std::cos(nRel) * (kRadarR - 7.f), 0.f);
            size_t bi = 0;
            for (auto& e : enemies) {
                if (!e->alive || bi >= blips.size()) continue;
                const PxVec3 ep = e->body->getGlobalPose().p;
                const float relX = ep.x - playerPos.x, relZ = ep.z - playerPos.z;
                const float d2 = relX * relX + relZ * relZ;
                const float rel = wrapPi(std::atan2(relX, relZ) - camYaw);
                const float r = std::min(std::sqrt(d2) / kRadarRange, 0.94f) * kRadarR;
                blips[bi]->position.set(-std::sin(rel) * r, std::cos(rel) * r, 0.f);
                blips[bi]->visible = true;
                ++bi;
            }
            for (; bi < blips.size(); ++bi) blips[bi]->visible = false;
        }

        vignette.material->opacity = std::clamp((60.f - health) / 60.f, 0.f, 0.55f);
        overDim.material->opacity += ((gameOver ? 0.55f : 0.f) - overDim.material->opacity) * std::min(1.f, dt * 6.f);
        for (auto& m : chMats) m->color = Color(hitMarkerT > 0.f ? kHudWarn : 0xffffff);

        // ===== render: world, then SVG overlay =====
        renderer->autoClear = true;
        renderer->render(*scene, *camera);
        renderer->autoClear = false;
        renderer->clearDepth();
        renderer->render(*ui, *uiCam);

        if (!shotPath.empty() && ++shotFrame >= shotFrames) {
            const auto path = fs::path(PROJECT_FOLDER) / "aaa_caps" / shotPath;
            if (auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get())) vk->writeFramebuffer(path);
            std::cout << "wrote " << path.string() << std::endl;
            std::exit(0);
        }
    });
}
