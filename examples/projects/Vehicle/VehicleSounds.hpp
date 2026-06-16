
#ifndef THREEPP_VEHICLE_SOUNDS_HPP
#define THREEPP_VEHICLE_SOUNDS_HPP

// Procedural audio rig for the PhysX vehicle demo. Same approach as the
// Shooter / Ocean examples: every loop and one-shot is synthesised once at
// startup, written to a temp WAV (the Audio API loads files), then driven
// each frame from PhysX telemetry:
//   • road    — low rumble ∝ ground speed while any wheel touches.
//   • wind    — gusty broadband that dominates at highway pace (∝ v²).
//   • thud    — suspension one-shot on compression-rate spikes (curbs, jumps).
//   • crash   — body impact one-shot on chassis Δv spikes (cones, rails).
//   • horn    — dual-tone loop while held; reverse beeper while backing up.
// (Engine and tire-squeal loops were tried and cut — synthesised versions
// didn't hold up against the rest. rpm()/skidLevel() telemetry stays for the
// HUD, and the hooks below are the place to drive recorded loops instead.)
// Loops are seamless: periodic components get an exact integer number of
// cycles over the loop length; noise/filter states crossfade over the seam.

#include "threepp/audio/Audio.hpp"
#include "threepp/cameras/Camera.hpp"
#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace threepp::vehiclesound {

    namespace detail {

        struct OnePole {
            float y = 0.f;
            float operator()(float x, float a) {
                y += a * (x - y);
                return y;
            }
        };
        inline float lpAlpha(float cutoffHz, int sr) {
            return 1.f - std::exp(-2.f * math::PI * cutoffHz / static_cast<float>(sr));
        }

        // Two-pole resonator (narrow bandpass). Noise through this is the
        // physically-honest way to get a squeal/ring: the band wanders in
        // phase and amplitude like a real resonance, where a raw sin() reads
        // as a flute. r in [0.98, 0.999] sets the Q.
        struct Resonator {
            float a1 = 0.f, a2 = 0.f, g = 1.f, y1 = 0.f, y2 = 0.f;
            void set(float freqHz, float r, int sr) {
                a1 = 2.f * r * std::cos(2.f * math::PI * freqHz / static_cast<float>(sr));
                a2 = -r * r;
                g  = 1.f - r;// rough gain normalisation
            }
            float operator()(float x) {
                const float y = g * x + a1 * y1 + a2 * y2;
                y2 = y1;
                y1 = y;
                return y;
            }
        };

        inline std::vector<float> normalized(std::vector<float> s, float peak) {
            float m = 0.f;
            for (float x : s) m = std::max(m, std::abs(x));
            if (m > 1e-6f)
                for (float& x : s) x *= peak / m;
            return s;
        }

        // Fold the `extra`-sample overhang back onto the head (linear crossfade)
        // so noise/filter state passes the loop seam without a click.
        inline std::vector<float> loopable(const std::vector<float>& s, int n, int extra) {
            std::vector<float> out(s.begin(), s.begin() + n);
            for (int i = 0; i < extra; ++i) {
                const float w = static_cast<float>(i) / static_cast<float>(extra);
                out[i] = s[n + i] * (1.f - w) + s[i] * w;
            }
            return out;
        }

        // 16-bit mono PCM WAV writer (verbatim from the Shooter example).
        inline void writeWav(const std::filesystem::path& path, const std::vector<float>& samples, int sr = 44100) {
            std::ofstream f(path, std::ios::binary);
            auto u32 = [&](std::uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); };
            auto u16 = [&](std::uint16_t v) { f.write(reinterpret_cast<char*>(&v), 2); };
            const auto dataBytes = static_cast<std::uint32_t>(samples.size()) * 2u;
            f.write("RIFF", 4);
            u32(36 + dataBytes);
            f.write("WAVE", 4);
            f.write("fmt ", 4);
            u32(16);
            u16(1);// PCM
            u16(1);// mono
            u32(sr);
            u32(sr * 2);
            u16(2);
            u16(16);
            f.write("data", 4);
            u32(dataBytes);
            for (float x : samples) {
                const auto q = static_cast<std::int16_t>(std::lround(std::clamp(x, -1.f, 1.f) * 32767.f));
                f.write(reinterpret_cast<const char*>(&q), 2);
            }
        }

        // Road/rolling noise, 2 s loop: deep low-passed rumble + a 13 Hz
        // texture flutter (26 exact cycles) that reads as seams/grain.
        inline std::vector<float> synthRoadLoop(int sr = 44100) {
            const float dur = 2.0f;
            const int n     = static_cast<int>(sr * dur);
            const int extra = sr / 4;
            std::mt19937 r(13);
            auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
            OnePole lpDeep, lpMid;
            const float aDeep = lpAlpha(110.f, sr);
            const float aMid  = lpAlpha(420.f, sr);
            std::vector<float> s(n + extra);
            for (int i = 0; i < n + extra; ++i) {
                const float t = static_cast<float>(i) / sr;
                const float w = rn();
                const float texture = 0.8f + 0.2f * std::sin(2.f * math::PI * 13.f * t);
                s[i] = lpDeep(w, aDeep) * 1.0f + lpMid(w, aMid) * 0.35f * texture;
            }
            return normalized(loopable(s, n, extra), 0.55f);
        }

        // Wind, 4 s loop — swept narrow band with 12 dB/oct edges and a hard
        // lull↔gust amplitude swing (same recipe as the Ocean demo, shorter
        // loop, gust LFOs loop-exact at k/4 Hz).
        inline std::vector<float> synthWindLoop(int sr = 44100) {
            const float dur = 4.0f;
            const int n     = static_cast<int>(sr * dur);
            const int extra = sr / 2;
            std::mt19937 r(17);
            auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
            OnePole hi1, hi2, lo1, lo2;
            std::vector<float> s(n + extra);
            for (int i = 0; i < n + extra; ++i) {
                const float t = static_cast<float>(i) / sr;
                float gust = 0.6f * std::sin(2.f * math::PI * 0.25f * t)
                           + 0.4f * std::sin(2.f * math::PI * 0.75f * t + 1.3f);
                gust = std::clamp(0.5f + 0.5f * gust, 0.f, 1.f);
                const float aHi = lpAlpha(220.f + 500.f * gust, sr);
                const float aLo = lpAlpha(70.f + 90.f * gust, sr);
                const float w   = rn();
                const float band = hi2(hi1(w, aHi), aHi) - lo2(lo1(w, aLo), aLo);
                s[i] = band * (0.15f + 0.85f * gust * gust);
            }
            return normalized(loopable(s, n, extra), 0.5f);
        }

        // Horn: two-tone (420 + 524 Hz — roughly the real dual-trumpet fourth),
        // each a band-limited SAWTOOTH (Σ sin(kf)/k) — the buzzy diaphragm
        // timbre. Plain sines here sounded like a recorder. 0.5 s loop, every
        // harmonic an integer number of cycles.
        inline std::vector<float> synthHornLoop(int sr = 44100) {
            const float dur = 0.5f;
            const int n     = static_cast<int>(sr * dur);
            std::vector<float> s(n);
            for (int i = 0; i < n; ++i) {
                const float t = static_cast<float>(i) / sr;
                float v = 0.f;
                for (const float f : {420.f, 524.f}) {
                    for (int k = 1; k <= 9; ++k) {
                        v += std::sin(2.f * math::PI * k * f * t) / static_cast<float>(k);
                    }
                }
                s[i] = v;
            }
            return normalized(std::move(s), 0.55f);
        }

        // Reverse beeper: 1 s loop, ~1 kHz tone on for 0.45 s with 10 ms edge
        // ramps. Square-ish (odd harmonics) like a real piezo alarm, not a
        // pure flute sine.
        inline std::vector<float> synthReverseLoop(int sr = 44100) {
            const float dur = 1.0f;
            const int n     = static_cast<int>(sr * dur);
            std::vector<float> s(n, 0.f);
            const int onN   = static_cast<int>(sr * 0.45f);
            const int ramp  = sr / 100;
            for (int i = 0; i < onN; ++i) {
                const float t   = static_cast<float>(i) / sr;
                float env = 1.f;
                if (i < ramp) env = static_cast<float>(i) / ramp;
                if (i > onN - ramp) env = static_cast<float>(onN - i) / ramp;
                const float v = std::sin(2.f * math::PI * 1000.f * t)
                              + std::sin(2.f * math::PI * 3000.f * t) * 0.33f
                              + std::sin(2.f * math::PI * 5000.f * t) * 0.20f;
                s[i] = v * env;
            }
            return normalized(std::move(s), 0.45f);
        }

        // Suspension thud one-shot — ALL noise, no oscillator (a decaying sine
        // is a cartoon "doink"). Heavy low-passed burst = the body thump, a
        // 200–700 Hz band = the bushing slap, a tiny wideband click on top.
        inline std::vector<float> synthThud(int sr = 44100) {
            const int n = static_cast<int>(sr * 0.25f);
            std::mt19937 r(19);
            auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
            OnePole lpBody, bsHi, bsLo;
            const float aBody = lpAlpha(130.f, sr);
            const float aHi   = lpAlpha(700.f, sr);
            const float aLo   = lpAlpha(200.f, sr);
            std::vector<float> s(n);
            for (int i = 0; i < n; ++i) {
                const float t = static_cast<float>(i) / sr;
                const float w = rn();
                s[i] = lpBody(w, aBody) * std::exp(-t * 26.f) * 2.6f
                     + (bsHi(w, aHi) - bsLo(w, aLo)) * std::exp(-t * 50.f) * 0.8f
                     + w * std::exp(-t * 240.f) * 0.25f;
            }
            return normalized(std::move(s), 0.8f);
        }

        // Body crash one-shot — layered broadband crunch, again no tonal
        // partials: low thump + a jagged mid crunch (noise gated by rectified
        // slow noise, so it crackles instead of whooshing) + a sparse debris
        // rattle riding two high resonators through the tail.
        inline std::vector<float> synthCrash(int sr = 44100) {
            const int n = static_cast<int>(sr * 0.6f);
            std::mt19937 r(23);
            auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
            auto u01 = [&] { return std::uniform_real_distribution<float>(0.f, 1.f)(r); };
            OnePole lpThump, crHi, crLo, lpGate;
            const float aThump = lpAlpha(110.f, sr);
            const float aCrHi  = lpAlpha(1900.f, sr);
            const float aCrLo  = lpAlpha(280.f, sr);
            const float aGate  = lpAlpha(55.f, sr);
            Resonator ring1, ring2;
            ring1.set(2300.f, 0.992f, sr);
            ring2.set(3400.f, 0.988f, sr);
            std::vector<float> s(n);
            for (int i = 0; i < n; ++i) {
                const float t = static_cast<float>(i) / sr;
                const float w = rn();
                const float gate   = std::clamp(0.2f + 2.5f * std::abs(lpGate(w, aGate)), 0.f, 1.f);
                const float thump  = lpThump(w, aThump) * std::exp(-t * 14.f) * 3.0f;
                const float crunch = (crHi(w, aCrHi) - crLo(w, aCrLo)) * gate * std::exp(-t * 8.f) * 1.1f;
                // Sparse impulses → ringing debris (bolts/trim rattling down).
                const float imp    = (u01() > 0.9985f) ? 1.f : 0.f;
                const float debris = (ring1(imp) + ring2(imp) * 0.7f) * std::exp(-t * 4.f) * 1.4f;
                s[i] = thump + crunch + debris;
            }
            return normalized(std::move(s), 0.85f);
        }

    }// namespace detail

    // Owns every source and drives them from PhysX telemetry once per frame.
    // Degrades to a no-op when no audio device is available.
    struct VehicleSounds {
        std::unique_ptr<AudioListener> listener;
        std::unique_ptr<Audio> road, wind, horn, reverseBeep, thud, crash;
        bool ok = false;

        // Telemetry kept for the HUD even though the engine/skid loops were
        // cut (and as ready-made drivers for recorded loops later).
        float rpm() const { return rpm_; }
        float skidLevel() const { return skid_; }

        void init() {
            namespace d = detail;
            try {
                const auto dir = std::filesystem::temp_directory_path() / "threepp_vehicle_sounds";
                std::filesystem::create_directories(dir);
                auto wav = [&](const char* name, const std::vector<float>& s) {
                    const auto p = dir / name;
                    d::writeWav(p, s);
                    return p;
                };
                const auto roadPath  = wav("road.wav", d::synthRoadLoop());
                const auto windPath  = wav("wind.wav", d::synthWindLoop());
                const auto hornPath  = wav("horn.wav", d::synthHornLoop());
                const auto revPath   = wav("reverse.wav", d::synthReverseLoop());
                const auto thudPath  = wav("thud.wav", d::synthThud());
                const auto crashPath = wav("crash.wav", d::synthCrash());

                listener = std::make_unique<AudioListener>();
                auto loop = [&](std::unique_ptr<Audio>& a, const std::filesystem::path& p) {
                    a = std::make_unique<Audio>(*listener, p);
                    a->setLooping(true);
                    a->setVolume(0.f);
                    a->play();
                };
                loop(road, roadPath);
                loop(wind, windPath);
                loop(horn, hornPath);
                loop(reverseBeep, revPath);
                thud = std::make_unique<Audio>(*listener, thudPath);
                crash = std::make_unique<Audio>(*listener, crashPath);
                ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[audio] disabled: " << e.what() << "\n";
            }
        }

        void update(float dt, const PhysxVehicle& vehicle, float throttle,
                    bool hornDown, const Camera& cam, float masterVolume) {
            if (!ok || dt <= 0.f) return;
            listener->setMasterVolume(masterVolume);

            const auto& s = vehicle.settings();
            const float fwdSpeed = vehicle.forwardSpeed();
            const float speedAbs = std::abs(fwdSpeed);

            // ── RPM proxy = driven-wheel spin (direct drive — wheelspin revs
            // it, locked brakes stall it), floored by a throttle blip. HUD
            // telemetry only since the engine loop was cut.
            float spin = 0.f;
            int nDriven = 0;
            for (int i = 0; i < 4; ++i) {
                if (s.drivenWheels[i]) {
                    spin += std::abs(vehicle.wheelAngularSpeed(i));
                    ++nDriven;
                }
            }
            const float spinMs = (nDriven ? spin / static_cast<float>(nDriven) : 0.f) * s.wheelRadius;
            float target = std::clamp(spinMs / 38.f, 0.f, 1.f);// ~38 m/s ≈ top speed
            target = std::max(target, 0.30f * throttle);
            const float tau = target > rpm_ ? 0.30f : 0.75f;// revs faster than it falls
            rpm_ += (target - rpm_) * (1.f - std::exp(-dt / tau));

            // ── Tire slip telemetry (HUD): worst grounded wheel beyond small
            // thresholds. Longitudinal slip is a ratio (1 = full wheelspin);
            // lateral is ≈ tan(slip angle) (0.12 ≈ 7°).
            float skidRaw = 0.f;
            bool anyGrounded = false;
            for (int i = 0; i < 4; ++i) {
                if (!vehicle.wheelGrounded(i)) continue;
                anyGrounded = true;
                const float lng = std::max(0.f, std::abs(vehicle.tireLongitudinalSlip(i)) - 0.20f);
                const float lat = std::max(0.f, std::abs(vehicle.tireLateralSlip(i)) - 0.12f);
                skidRaw = std::max(skidRaw, lng * 1.2f + lat * 2.5f);
            }
            if (speedAbs < 1.5f && spinMs < 3.f) skidRaw = 0.f;
            skidRaw = std::clamp(skidRaw, 0.f, 1.f);
            const float skidTau = skidRaw > skid_ ? 0.06f : 0.22f;// attack fast, release smooth
            skid_ += (skidRaw - skid_) * (1.f - std::exp(-dt / skidTau));

            // ── Road rumble + wind.
            const float roadNorm = std::clamp(speedAbs / 25.f, 0.f, 1.f);
            road->setVolume(anyGrounded ? 0.55f * roadNorm : 0.f);
            road->setPlaybackRate(0.8f + 0.5f * roadNorm);
            const float windNorm = std::clamp(speedAbs / 45.f, 0.f, 1.f);
            wind->setVolume(windNorm * windNorm * 0.8f);

            // ── Suspension thuds: per-wheel compression-rate spike + cooldown.
            float worstJounce = 0.f;
            for (int i = 0; i < 4; ++i) {
                thudCooldown_[i] = std::max(0.f, thudCooldown_[i] - dt);
                const float js = std::abs(vehicle.suspensionJounceSpeed(i));
                if (vehicle.wheelGrounded(i) && js > 0.8f && thudCooldown_[i] <= 0.f) {
                    worstJounce = std::max(worstJounce, js);
                    thudCooldown_[i] = 0.20f;
                }
            }
            if (worstJounce > 0.f) {
                thud->setVolume(std::clamp(worstJounce / 3.f, 0.25f, 1.f) * 0.8f);
                // Tiny deterministic pitch variation so repeated hits don't machine-gun.
                thudPitchFlip_ = !thudPitchFlip_;
                thud->setPlaybackRate(thudPitchFlip_ ? 0.92f : 1.08f);
                thud->seekToStart();
                thud->play();
            }

            // ── Crash: chassis Δv spike well beyond what tires can produce
            // (full braking is ~2 g; rails and cone walls are not).
            crashCooldown_ = std::max(0.f, crashCooldown_ - dt);
            Vector3 vel;
            {
                const auto v = vehicle.chassisActor()->getLinearVelocity();
                vel.set(v.x, v.y, v.z);
            }
            if (haveVel_) {
                const float jerk = vel.distanceTo(prevVel_) / dt;// m/s²
                if (jerk > 60.f && crashCooldown_ <= 0.f) {
                    crash->setVolume(std::clamp(jerk / 300.f, 0.3f, 1.f));
                    crash->setPlaybackRate(0.9f + 0.2f * (thudPitchFlip_ ? 1.f : 0.f));
                    crash->seekToStart();
                    crash->play();
                    crashCooldown_ = 0.5f;
                }
            }
            prevVel_ = vel;
            haveVel_ = true;

            // ── Horn + reverse beeper (volume-gated loops — no click on edges).
            horn->setVolume(hornDown ? 0.8f : 0.f);
            const bool backing = vehicle.gear() == PhysxVehicle::Gear::Reverse && speedAbs > 0.3f;
            reverseBeep->setVolume(backing ? 0.4f : 0.f);

            // ── Listener rides the active camera.
            listener->position.copy(cam.position);
            listener->quaternion.copy(cam.quaternion);
            listener->updateMatrixWorld(true);
        }

    private:
        float rpm_ = 0.f;
        float skid_ = 0.f;
        std::array<float, 4> thudCooldown_{};
        float crashCooldown_ = 0.f;
        Vector3 prevVel_;
        bool haveVel_ = false;
        bool thudPitchFlip_ = false;
    };

}// namespace threepp::vehiclesound

#endif//THREEPP_VEHICLE_SOUNDS_HPP
