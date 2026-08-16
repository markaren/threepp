// Horizontal 360-degree sector scan feeding OBSTACLE_DISTANCE. This is the
// frame/sector arithmetic only — the ray backend is injected, so the geometry
// unit-tests without PhysX and the same code serves a PhysX scene, a heightmap
// probe, or a synthetic caster in a test.
//
// Frames follow include/threepp/extras/uav/FrameConv.hpp, the single frame
// authority in this tree: threepp world is Y-up with North = -Z and East = +X.
// Nothing here is allowed to invent a second convention.
//
// Two things this gets deliberately right, both of which are how proximity
// feeds usually fail:
//
//  1. Rays are cast HORIZONTALLY IN THE WORLD FRAME, yaw-aligned only. A quad
//     pitched 25 degrees forward to accelerate would otherwise scan its own
//     fan into the ground and report a wall of obstacles ahead of itself.
//  2. Free space is reported as a MEASUREMENT, not as UNKNOWN. UINT16_MAX
//     means "no data" in the spec, and a planner given no data will not plan
//     through the gap; our rays genuinely measured that space as empty, so a
//     clear sector reports max+1, the spec's "beyond max range" idiom.

#ifndef THREEPP_EXTRAS_UAV_PROXIMITYSCAN_HPP
#define THREEPP_EXTRAS_UAV_PROXIMITYSCAN_HPP

#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>

namespace threepp::uav {

    /// (originWorld, dirWorld, maxRange) -> hit distance in metres, <0 = miss.
    /// dirWorld arrives normalized.
    using RayCaster = std::function<float(const Vector3&, const Vector3&, float)>;

    class ProximityScan {

    public:
        static constexpr std::size_t sectors = 72;
        using Distances = std::array<std::uint16_t, sectors>;

        /// The spec's "no data" value. We never emit it — see the file header.
        static constexpr std::uint16_t unknown = 65535;

        struct Params {
            float incrementDeg = 5.f; ///< 360 / sectors; also the wire field
            float minRange = 0.30f;   ///< reported as min_distance
            float maxRange = 50.f;    ///< reported as max_distance
            int raysPerSector = 3;    ///< min of these wins the sector
            float raySpreadDeg = 1.7f;///< half-spread about the sector centre
        };

        explicit ProximityScan(const Params& p = {}): params_(p) {
            distances_.fill(clearValue());
        }

        [[nodiscard]] const Params& params() const { return params_; }

        [[nodiscard]] std::uint16_t maxCm() const {
            return static_cast<std::uint16_t>(params_.maxRange * 100.f);
        }
        [[nodiscard]] std::uint16_t minCm() const {
            return static_cast<std::uint16_t>(params_.minRange * 100.f);
        }
        /// Measured free space: one past max range, NOT `unknown`.
        [[nodiscard]] std::uint16_t clearValue() const {
            return static_cast<std::uint16_t>(maxCm() + 1);
        }

        /// Yaw of the last scan [rad], clockwise from North (0 = nose North).
        [[nodiscard]] float yaw() const { return yaw_; }

        [[nodiscard]] const Distances& distances() const { return distances_; }

        /// World direction of sector k's centre at the last scan's yaw.
        [[nodiscard]] Vector3 sectorDir(std::size_t k) const {
            return dirFromAzimuth(yaw_ + sectorCentreOffset(k));
        }

        /// Bearing of sector k relative to the nose [deg], sector centre.
        [[nodiscard]] float sectorBearingDeg(std::size_t k) const {
            return static_cast<float>(k) * params_.incrementDeg + 0.5f * params_.incrementDeg;
        }

        /// Fill `distances()` from one 360-degree sweep about `positionWorld`.
        /// Sector 0 straddles the nose and sectors advance CLOCKWISE seen from
        /// above (MAV_FRAME_BODY_FRD).
        const Distances& scan(const RayCaster& cast, const Vector3& positionWorld,
                              const Vector3& forwardWorld) {
            yaw_ = yawFromForward(forwardWorld, yaw_);

            const int rays = std::max(1, params_.raysPerSector);
            const float spread = deg2rad(params_.raySpreadDeg);
            const float clear = static_cast<float>(clearValue());
            const float maxCmF = static_cast<float>(maxCm());

            for (std::size_t k = 0; k < sectors; ++k) {
                const float centre = yaw_ + sectorCentreOffset(k);
                float best = -1.f;
                for (int i = 0; i < rays; ++i) {
                    // Ray i spans [-spread, +spread]; odd counts include the
                    // centre. A 0.3 m trunk must not slip between the rays of a
                    // 5-degree sector, which is what the spread buys.
                    const float t = rays == 1 ? 0.f
                                              : -spread + 2.f * spread * static_cast<float>(i) /
                                                                  static_cast<float>(rays - 1);
                    const float d = cast(positionWorld, dirFromAzimuth(centre + t), params_.maxRange);
                    if (d >= 0.f && (best < 0.f || d < best)) best = d;
                }
                // A hit closer than minRange is still a hit — reporting it as
                // clear is how you fly into the thing you are touching.
                distances_[k] = best < 0.f
                                        ? static_cast<std::uint16_t>(clear)
                                        : static_cast<std::uint16_t>(
                                                  std::clamp(best * 100.f, 0.f, maxCmF));
            }
            return distances_;
        }

        /// Azimuth measured CLOCKWISE from North -> threepp world direction.
        /// a = 0 gives -Z (North); a = pi/2 gives +X (East). See FrameConv.
        static Vector3 dirFromAzimuth(float a) {
            return {std::sin(a), 0.f, -std::cos(a)};
        }

        /// Yaw from a body-forward vector, projected onto the ground plane.
        /// A degenerate (purely vertical) forward keeps `fallback` — a quad
        /// pointed straight up has no meaningful heading, and snapping the fan
        /// to North for one frame would flash a false obstacle ring.
        static float yawFromForward(const Vector3& forwardWorld, float fallback = 0.f) {
            const float x = forwardWorld.x, z = forwardWorld.z;
            if (x * x + z * z < 1e-8f) return fallback;
            return std::atan2(x, -z);
        }

    private:
        [[nodiscard]] float sectorCentreOffset(std::size_t k) const {
            return deg2rad((static_cast<float>(k) + 0.5f) * params_.incrementDeg);
        }

        static float deg2rad(float d) {
            return d * 0.017453292519943295f;
        }

        Params params_;
        Distances distances_{};
        float yaw_ = 0.f;
    };

}// namespace threepp::uav

#endif// THREEPP_EXTRAS_UAV_PROXIMITYSCAN_HPP
