// ArduPilot SITL "JSON" physics-backend bridge.
//
// SITL (launched with `-f JSON:<our-ip>`) sends a binary servo packet to UDP
// port 9002 every physics frame and then BLOCKS until we reply with one
// newline-terminated JSON line of vehicle state. Our reply timestamp drives
// SITL's clock (lock-step): we are the simulation's timebase, SITL is the
// autopilot. Replies go to whatever address sent the last packet, which is why
// this works through the WSL2 NAT with zero configuration — the reply rides
// the hole the inbound packet punched.
//
// Wire format (little-endian, both sides x86):
//   uint16 magic       18458 => 16 pwm channels (40-byte packet)
//                      29569 => 32 pwm channels (72-byte packet, SERVO_32_ENABLE)
//   uint16 frame_rate  declared physics rate, Hz
//   uint32 frame_count monotonically increasing; a decrease means SITL restarted
//   uint16 pwm[N]      servo outputs, microseconds (~1000..2000)
//
// The socket lives behind the Impl in SitlBridge.cpp, so this header pulls in
// no platform headers (winsock must not leak into every consumer). State
// in/out is plain doubles in ArduPilot's own frames (NED world, FRD body).

#ifndef THREEPP_EXTRAS_UAV_SITLBRIDGE_HPP
#define THREEPP_EXTRAS_UAV_SITLBRIDGE_HPP

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

namespace threepp::uav {

    /// One decoded servo packet from SITL.
    struct ServoInput {
        std::uint16_t frameRate = 0;///< declared physics rate [Hz]
        std::uint32_t frameCount = 0;
        std::uint16_t pwm[32] = {};///< [µs]; channels beyond `channels` stay 0
        int channels = 0;          ///< 16 or 32
    };

    /// Vehicle state for one physics frame, in ArduPilot conventions:
    /// NED world frame, FRD (x fwd, y right, z down) body frame, SI units.
    /// Optional fields are omitted from the JSON while NaN.
    struct FdmState {
        double timestampSec = 0; ///< absolute sim time; drives SITL's clock
        double gyro[3] = {};     ///< body FRD [rad/s]
        double accelBody[3] = {};///< specific force, body FRD [m/s^2]
        double positionNed[3] = {};
        double velocityNed[3] = {};
        double attitudeRpy[3] = {};///< roll, pitch, yaw [rad]
        double rangefinderM = NAN; ///< downward AGL -> "rng_1"
        double airspeed = NAN;     ///< |v - wind| [m/s]
        double windNed[3] = {NAN, 0, 0};///< -> "velocity_wind"; NaN first = omit
        double batteryV = NAN, batteryA = NAN;
    };

    /// UDP endpoint + protocol codec. poll() is non-blocking; call it until it
    /// returns None, replying to every Frame with sendState() to keep lock-step.
    class SitlBridge {
    public:
        enum class Event {
            None, ///< socket drained
            Frame,///< `out` holds a servo frame that must be answered
            Reset ///< SITL restarted (frame_count went backwards / new peer);
                  ///< `out` holds the first frame of the new run — answer it too
        };

        /// Binds the given UDP port on all interfaces. Port 0 picks an
        /// ephemeral port (see port()) — used by selftests to avoid clashing
        /// with a real SITL session. Handles winsock startup on Windows.
        explicit SitlBridge(std::uint16_t port = 9002);

        ~SitlBridge();

        SitlBridge(const SitlBridge&) = delete;
        SitlBridge& operator=(const SitlBridge&) = delete;

        /// False if the constructor failed to bind (port in use, no Winsock).
        [[nodiscard]] bool valid() const;

        /// The actually-bound port (differs from the request only for port 0).
        [[nodiscard]] std::uint16_t port() const;

        Event poll(ServoInput& out);

        /// Format the state as one JSON line and send it to the last peer.
        void sendState(const FdmState& s);

        [[nodiscard]] bool connected() const;
        [[nodiscard]] std::uint32_t lastFrameCount() const;

        /// "a.b.c.d:port" of the SITL instance, or "" before first contact.
        [[nodiscard]] std::string peer() const;

        /// EMA of the achieved frame rate over wall time, for HUD display.
        [[nodiscard]] double achievedRateHz() const;

        /// Decode a raw datagram. Static so tests hit it directly.
        static bool decode(const std::uint8_t* buf, int n, ServoInput& out);

        /// Encode a servo packet (fake-SITL selftests use this).
        static int encode(const ServoInput& in, std::uint8_t (&buf)[128]);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}// namespace threepp::uav

#endif// THREEPP_EXTRAS_UAV_SITLBRIDGE_HPP
