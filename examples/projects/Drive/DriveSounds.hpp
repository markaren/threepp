
#ifndef THREEPP_DRIVE_SOUNDS_HPP
#define THREEPP_DRIVE_SOUNDS_HPP

// Procedural audio for the engine-drive driving demo. Reuses the synthesis
// toolkit (filters, WAV writer, road/wind/horn/reverse/thud/crash loops) from
// the existing Vehicle example's VehicleSounds, and adds the one thing the
// direct-drive demo couldn't do honestly: an ENGINE drone pitched by the real
// PxVehicle2 engine speed. The engine loop is a band-limited buzz synthesised
// once at idle pitch, then resampled per frame by (rpm / idleRpm) so the note
// climbs and falls with the tacho — plus a clutch-slip flare on gear changes.

#include "../Vehicle/VehicleSounds.hpp"// reuses threepp::vehiclesound::detail

#include "threepp/extras/physx/PhysxVehicleEngineDrive.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace drive {

    using namespace threepp;

    namespace detail {

        // Band-limited engine buzz, loop-exact at f0 = 100 Hz over 0.2 s (20 cycles).
        // A saw-ish stack of harmonics (uneven weights = lumpy combustion order) plus
        // a touch of low-passed noise for grit. Pitched at runtime via playback rate.
        inline std::vector<float> synthEngineLoop(int sr = 44100) {
            namespace vd = threepp::vehiclesound::detail;
            const float f0 = 100.f;
            const float dur = 0.2f;
            const int n = static_cast<int>(sr * dur);
            const int extra = sr / 20;
            std::mt19937 r(29);
            auto rn = [&] { return std::uniform_real_distribution<float>(-1.f, 1.f)(r); };
            vd::OnePole lp;
            const float a = vd::lpAlpha(900.f, sr);
            std::vector<float> s(n + extra);
            // Per-harmonic weights — emphasise the low orders for a torquey idle.
            const float w[8] = {1.0f, 0.85f, 0.5f, 0.62f, 0.3f, 0.22f, 0.16f, 0.1f};
            for (int i = 0; i < n + extra; ++i) {
                const float t = static_cast<float>(i) / sr;
                float v = 0.f;
                for (int k = 1; k <= 8; ++k)
                    v += w[k - 1] * std::sin(2.f * math::PI * static_cast<float>(k) * f0 * t) / static_cast<float>(k);
                v += lp(rn(), a) * 0.25f;// combustion grit
                s[i] = v;
            }
            return vd::normalized(vd::loopable(s, n, extra), 0.6f);
        }

    }// namespace detail

    // Owns every source and drives them from PxVehicle2 telemetry once per frame.
    // Degrades to a no-op when no audio device is available.
    struct DriveSounds {
        std::unique_ptr<AudioListener> listener;
        std::unique_ptr<Audio> road, wind, engine, reverseBeep, horn, thud, crash;
        bool ok = false;

        float rpmFraction() const { return rpmFrac_; }
        float skidLevel() const { return skid_; }

        void init() {
            namespace vd = threepp::vehiclesound::detail;
            try {
                const auto dir = std::filesystem::temp_directory_path() / "threepp_drive_sounds";
                std::filesystem::create_directories(dir);
                auto wav = [&](const char* name, const std::vector<float>& s) {
                    const auto p = dir / name;
                    vd::writeWav(p, s);
                    return p;
                };
                const auto roadPath = wav("road.wav", vd::synthRoadLoop());
                const auto windPath = wav("wind.wav", vd::synthWindLoop());
                const auto engPath = wav("engine.wav", detail::synthEngineLoop());
                const auto revPath = wav("reverse.wav", vd::synthReverseLoop());
                const auto hornPath = wav("horn.wav", vd::synthHornLoop());
                const auto thudPath = wav("thud.wav", vd::synthThud());
                const auto crashPath = wav("crash.wav", vd::synthCrash());

                listener = std::make_unique<AudioListener>();
                auto loop = [&](std::unique_ptr<Audio>& a, const std::filesystem::path& p) {
                    a = std::make_unique<Audio>(*listener, p);
                    a->setLooping(true);
                    a->setVolume(0.f);
                    a->play();
                };
                loop(road, roadPath);
                loop(wind, windPath);
                loop(engine, engPath);
                loop(reverseBeep, revPath);
                loop(horn, hornPath);
                thud = std::make_unique<Audio>(*listener, thudPath);
                crash = std::make_unique<Audio>(*listener, crashPath);
                ok = true;
            } catch (const std::exception& e) {
                std::cerr << "[audio] disabled: " << e.what() << "\n";
            }
        }

        void update(float dt, const PhysxVehicleEngineDrive& vehicle, float throttle,
                    bool hornDown, const Camera& cam, float masterVolume) {
            if (!ok || dt <= 0.f) return;
            listener->setMasterVolume(masterVolume);

            const auto& s = vehicle.settings();
            const float speedAbs = std::abs(vehicle.forwardSpeed());

            // ── Engine: pitch tracks real engine revs; volume blends idle bed +
            //    load (throttle) + revs. A clutch-slip flare adds a moment of
            //    flutter on gear changes.
            const float rpm = vehicle.engineRpm();
            rpmFrac_ += (vehicle.engineRpmFraction() - rpmFrac_) * (1.f - std::exp(-dt / 0.08f));
            const float idle = std::max(vehicle.engineIdleRpm(), 1.f);
            // Loop was synthesised at 100 Hz ≈ idle; map revs onto playback rate.
            const float rate = std::clamp(rpm / idle, 0.5f, 7.f);
            engine->setPlaybackRate(rate);
            const float load = std::clamp(0.35f + 0.65f * throttle, 0.f, 1.f);
            engine->setVolume((0.22f + 0.5f * rpmFrac_) * load * 0.9f);

            // ── RPM/skid HUD telemetry.
            float skidRaw = 0.f;
            bool anyGrounded = false;
            for (int i = 0; i < 4; ++i) {
                if (!vehicle.wheelGrounded(i)) continue;
                anyGrounded = true;
                const float lng = std::max(0.f, std::abs(vehicle.tireLongitudinalSlip(i)) - 0.20f);
                const float lat = std::max(0.f, std::abs(vehicle.tireLateralSlip(i)) - 0.12f);
                skidRaw = std::max(skidRaw, lng * 1.2f + lat * 2.5f);
            }
            if (speedAbs < 1.5f) skidRaw = 0.f;
            skidRaw = std::clamp(skidRaw, 0.f, 1.f);
            const float skidTau = skidRaw > skid_ ? 0.06f : 0.22f;
            skid_ += (skidRaw - skid_) * (1.f - std::exp(-dt / skidTau));

            // ── Road rumble + wind.
            const float roadNorm = std::clamp(speedAbs / 25.f, 0.f, 1.f);
            road->setVolume(anyGrounded ? 0.5f * roadNorm : 0.f);
            road->setPlaybackRate(0.8f + 0.5f * roadNorm);
            const float windNorm = std::clamp(speedAbs / 45.f, 0.f, 1.f);
            wind->setVolume(windNorm * windNorm * 0.8f);

            // ── Suspension thuds on compression-rate spikes.
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
                thudPitchFlip_ = !thudPitchFlip_;
                thud->setPlaybackRate(thudPitchFlip_ ? 0.92f : 1.08f);
                thud->seekToStart();
                thud->play();
            }

            // ── Crash on chassis Δv spikes.
            crashCooldown_ = std::max(0.f, crashCooldown_ - dt);
            Vector3 vel;
            {
                const auto v = vehicle.chassisActor()->getLinearVelocity();
                vel.set(v.x, v.y, v.z);
            }
            if (haveVel_) {
                const float jerk = vel.distanceTo(prevVel_) / dt;
                if (jerk > 60.f && crashCooldown_ <= 0.f) {
                    crash->setVolume(std::clamp(jerk / 300.f, 0.3f, 1.f));
                    crash->seekToStart();
                    crash->play();
                    crashCooldown_ = 0.5f;
                }
            }
            prevVel_ = vel;
            haveVel_ = true;

            // ── Horn + reverse beeper.
            horn->setVolume(hornDown ? 0.8f : 0.f);
            const bool backing = vehicle.direction() == PhysxVehicleEngineDrive::Direction::Reverse && speedAbs > 0.3f;
            reverseBeep->setVolume(backing ? 0.4f : 0.f);

            listener->position.copy(cam.position);
            listener->quaternion.copy(cam.quaternion);
            listener->updateMatrixWorld(true);
        }

    private:
        float rpmFrac_ = 0.f;
        float skid_ = 0.f;
        std::array<float, 4> thudCooldown_{};
        float crashCooldown_ = 0.f;
        Vector3 prevVel_;
        bool haveVel_ = false;
        bool thudPitchFlip_ = false;
    };

}// namespace drive

#endif//THREEPP_DRIVE_SOUNDS_HPP
