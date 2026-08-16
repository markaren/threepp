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
//
// Two ways to fill the sectors, same arithmetic behind both:
//
//   scan()                         cast a fan of rays, one sweep per call.
//   beginFrame() + feed()...       bin an existing point cloud (a real LIDAR).
//
// The second exists because a ray fan against a physics scene only sees what
// has a collider, and a simulated canopy usually has none — the vehicle flies
// into leaves the camera plainly draws. A traced LIDAR sees what the renderer
// sees. What the cloud path CANNOT do is cast horizontally the way (1) does:
// the returns arrive from wherever the sensor's rings pointed, including
// straight at the ground five metres below. That is what the elevation slab
// replaces the yaw-frame-casting trick with — see Params::slabBelow. The slab
// rides the sensor, so it stops holding the ground out once the sensor is
// close to it; below about a metre AGL the floor is inside the slab. That case
// gets the answer a perception stack gives — segment the ground and drop it —
// via beginFrame()'s optional groundY hint.

#ifndef THREEPP_EXTRAS_UAV_PROXIMITYSCAN_HPP
#define THREEPP_EXTRAS_UAV_PROXIMITYSCAN_HPP

#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

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
            // Elevation slab for the cloud path (feed()), relative to the
            // sensor. A return outside it is not an obstacle THIS vehicle can
            // hit at THIS height: the ground under the drone and the canopy
            // arching overhead both fall out, and neither becomes a phantom
            // wall in some sector. Asymmetric on purpose — a quad has more to
            // fear from the branch it is about to climb into than from the
            // field it is about to leave.
            float slabBelow = -1.0f;  ///< metres below the sensor, inclusive
            float slabAbove = 2.5f;   ///< metres above the sensor, inclusive
            // Ground segmentation, the other half of the cloud path's filter.
            // The slab rides the SENSOR, so on a landing approach the ground
            // climbs into it: below about a metre AGL every downward ring
            // reports the floor as an in-slab return and all 72 sectors read a
            // sub-metre wall — the autopilot then flies avoidance against the
            // pad it is trying to touch. A perception stack answers this by
            // labelling the ground plane and dropping it, which is exactly
            // what beginFrame()'s groundY hint plus this margin do. The margin
            // is what makes it segmentation rather than a plane subtraction:
            // grass, gravel spray and the terrain's own centimetres of relief
            // all sit above the fitted plane and none of them is an obstacle.
            float groundClearance = 0.5f;///< metres above groundY that still reads as ground
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

        /// Open a cloud-fed sweep: every sector back to "measured clear", the
        /// yaw latched once so all of this frame's returns bin against ONE
        /// heading. Follow with feed() per return; distances() is valid at any
        /// point after (a half-filled frame reads as partly clear, never as
        /// stale — which is the honest answer while the returns are arriving).
        ///
        /// `groundY` is the world-Y of the terrain under the vehicle — from a
        /// downward altimeter, a heightmap query, whatever the caller trusts.
        /// Finite values switch ground segmentation on for this frame (see
        /// Params::groundClearance); NaN, the default, means "unknown" and
        /// leaves the slab as the only height filter. Unknown deliberately
        /// reads as the OLD behaviour rather than as ground at y=0: a hint
        /// invented for a vehicle over a hillside would delete the obstacles
        /// on the uphill side.
        void beginFrame(const Vector3& positionWorld, const Vector3& forwardWorld,
                        float groundY = std::numeric_limits<float>::quiet_NaN()) {
            yaw_ = yawFromForward(forwardWorld, yaw_);
            origin_ = positionWorld;
            groundY_ = groundY;
            distances_.fill(clearValue());
        }

        /// Bin one world-space LIDAR return into its sector. Nearest wins.
        void feed(const Vector3& hitWorld) {
            const float dy = hitWorld.y - origin_.y;
            if (dy < params_.slabBelow || dy > params_.slabAbove) return;

            // AND with the slab, not instead of it: the slab bounds what this
            // vehicle can hit at this height, the ground plane bounds what is
            // an obstacle at all. On descent the two disagree — the floor is
            // inside the slab and is not an obstacle — and that disagreement
            // is the phantom wall this rejects.
            if (std::isfinite(groundY_) && hitWorld.y < groundY_ + params_.groundClearance) return;

            const float dx = hitWorld.x - origin_.x;
            const float dz = hitWorld.z - origin_.z;
            const float horiz = std::sqrt(dx * dx + dz * dz);
            // Straight up or straight down has no bearing to bin it into, and
            // atan2(0, 0) would silently hand it to whichever sector rounds
            // first. It is also already inside the vehicle's own footprint.
            if (horiz < 1e-4f) return;
            // Past our declared max range is not an obstacle we report — it is
            // the free space clearValue() already says it is. Clamping it in
            // would plant a fake wall at exactly max range in every sector the
            // sensor happens to out-range us in.
            if (horiz > params_.maxRange) return;

            // The sector the return's bearing FALLS IN, not the nearest sector
            // centre: sector k spans [k, k+1) increments clockwise from the
            // nose, exactly the span scan()'s rays sweep.
            const float rel = std::atan2(dx, -dz) - yaw_;
            const auto n = static_cast<int>(sectors);
            int k = static_cast<int>(std::floor(rel / deg2rad(params_.incrementDeg))) % n;
            if (k < 0) k += n;

            // HORIZONTAL distance, not slant range: OBSTACLE_DISTANCE describes
            // a horizontal fan, and a return 2 m up at 10 m out is a thing 10 m
            // ahead — reporting its 10.2 m slant would have the planner brake
            // slightly late, every time, in the direction that matters.
            const auto cm = static_cast<std::uint16_t>(
                    std::clamp(horiz * 100.f, 0.f, static_cast<float>(maxCm())));
            if (cm < distances_[static_cast<std::size_t>(k)]) {
                distances_[static_cast<std::size_t>(k)] = cm;
            }
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
        Vector3 origin_;///< sensor position of the open cloud frame (feed())
        /// Ground hint of the open cloud frame; NaN = unknown, see beginFrame.
        float groundY_ = std::numeric_limits<float>::quiet_NaN();
    };

}// namespace threepp::uav

#endif// THREEPP_EXTRAS_UAV_PROXIMITYSCAN_HPP
