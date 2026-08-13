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
// This header is deliberately free of threepp includes so the editor's play
// runtime can adopt it unchanged later; state in/out is plain doubles in
// ArduPilot's own frames (NED world, FRD body).

#ifndef THREEPP_EXAMPLE_SITL_BRIDGE_HPP
#define THREEPP_EXAMPLE_SITL_BRIDGE_HPP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX// windows.h min/max macros break PhysX and <algorithm>
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
// windef.h's 16-bit relics; threepp cameras have members named near/far.
#undef near
#undef far
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace sitl {

#ifdef _WIN32
    using socket_t = SOCKET;
    inline constexpr socket_t invalidSocket = INVALID_SOCKET;
#else
    using socket_t = int;
    inline constexpr socket_t invalidSocket = -1;
    inline void closesocket(socket_t s) { ::close(s); }
#endif

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
        /// ephemeral port (see port()) — used by the selftest to avoid
        /// clashing with a real SITL session.
        explicit SitlBridge(std::uint16_t port = 9002) {
#ifdef _WIN32
            static const int wsaInit = [] {
                WSADATA data;
                return WSAStartup(MAKEWORD(2, 2), &data);
            }();
            (void) wsaInit;
#endif
            sock_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            if (sock_ == invalidSocket) return;

            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
            addr.sin_port = htons(port);
            if (::bind(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
                closesocket(sock_);
                sock_ = invalidSocket;
                return;
            }
#ifdef _WIN32
            u_long nonBlocking = 1;
            ioctlsocket(sock_, FIONBIO, &nonBlocking);
#else
            fcntl(sock_, F_SETFL, fcntl(sock_, F_GETFL, 0) | O_NONBLOCK);
#endif
            sockaddr_in bound{};
#ifdef _WIN32
            int len = sizeof bound;
#else
            socklen_t len = sizeof bound;
#endif
            if (::getsockname(sock_, reinterpret_cast<sockaddr*>(&bound), &len) == 0) {
                port_ = ntohs(bound.sin_port);
            }
        }

        ~SitlBridge() {
            if (sock_ != invalidSocket) closesocket(sock_);
        }

        SitlBridge(const SitlBridge&) = delete;
        SitlBridge& operator=(const SitlBridge&) = delete;

        /// False if the constructor failed to bind (port in use, no Winsock).
        [[nodiscard]] bool valid() const { return sock_ != invalidSocket; }

        /// The actually-bound port (differs from the request only for port 0).
        [[nodiscard]] std::uint16_t port() const { return port_; }

        Event poll(ServoInput& out) {
            if (sock_ == invalidSocket) return Event::None;

            std::uint8_t buf[128];
            sockaddr_in from{};
#ifdef _WIN32
            int fromLen = sizeof from;
            const int n = ::recvfrom(sock_, reinterpret_cast<char*>(buf), sizeof buf, 0,
                                     reinterpret_cast<sockaddr*>(&from), &fromLen);
#else
            socklen_t fromLen = sizeof from;
            const auto n = ::recvfrom(sock_, buf, sizeof buf, 0,
                                      reinterpret_cast<sockaddr*>(&from), &fromLen);
#endif
            if (n <= 0) return Event::None;

            if (!decode(buf, static_cast<int>(n), out)) {
                if (!warnedBadPacket_) {
                    std::fprintf(stderr, "[sitl] dropping unrecognized %d-byte packet (bad magic/length)\n",
                                 static_cast<int>(n));
                    warnedBadPacket_ = true;
                }
                return Event::None;
            }

            const bool newPeer = from.sin_addr.s_addr != peer_.sin_addr.s_addr ||
                                 from.sin_port != peer_.sin_port;
            const sockaddr_in prevPeer = peer_;
            peer_ = from;
            connected_ = true;
            trackRate();

            // frame_count going backwards means SITL restarted; SITL also
            // re-sends the SAME frame every 10 s while unanswered, so equality
            // is a retry, not a restart.
            const bool restarted = (newPeer && haveFrame_) ||
                                   (haveFrame_ && out.frameCount < lastFrameCount_);
            if (restarted) {
                // Name the trigger: an alternating peer means TWO SITL
                // instances are feeding this port — a reset storm that looks
                // like endless restarts. Kill the stray one.
                char oldIp[INET_ADDRSTRLEN] = {}, newIp[INET_ADDRSTRLEN] = {};
                inet_ntop(AF_INET, &prevPeer.sin_addr, oldIp, sizeof oldIp);
                inet_ntop(AF_INET, &from.sin_addr, newIp, sizeof newIp);
                std::fprintf(stderr,
                             "[sitl] reset: %s (peer %s:%u -> %s:%u, frame %u -> %u)\n",
                             newPeer ? "PEER CHANGED" : "frame_count rollback",
                             oldIp, ntohs(prevPeer.sin_port), newIp, ntohs(from.sin_port),
                             lastFrameCount_, out.frameCount);
            }
            haveFrame_ = true;
            lastFrameCount_ = out.frameCount;
            return restarted ? Event::Reset : Event::Frame;
        }

        /// Format the state as one JSON line and send it to the last peer.
        void sendState(const FdmState& s) {
            if (sock_ == invalidSocket || !connected_) return;

            char json[512];
            int len = std::snprintf(
                    json, sizeof json,
                    "{\"timestamp\":%.9f,"
                    "\"imu\":{\"gyro\":[%.9f,%.9f,%.9f],\"accel_body\":[%.9f,%.9f,%.9f]},"
                    "\"position\":[%.9f,%.9f,%.9f],"
                    "\"velocity\":[%.9f,%.9f,%.9f],"
                    "\"attitude\":[%.9f,%.9f,%.9f]",
                    s.timestampSec,
                    s.gyro[0], s.gyro[1], s.gyro[2],
                    s.accelBody[0], s.accelBody[1], s.accelBody[2],
                    s.positionNed[0], s.positionNed[1], s.positionNed[2],
                    s.velocityNed[0], s.velocityNed[1], s.velocityNed[2],
                    s.attitudeRpy[0], s.attitudeRpy[1], s.attitudeRpy[2]);
            if (!std::isnan(s.rangefinderM)) {
                len += std::snprintf(json + len, sizeof json - len, ",\"rng_1\":%.3f", s.rangefinderM);
            }
            if (!std::isnan(s.airspeed)) {
                len += std::snprintf(json + len, sizeof json - len, ",\"airspeed\":%.3f", s.airspeed);
            }
            if (!std::isnan(s.windNed[0])) {
                len += std::snprintf(json + len, sizeof json - len,
                                     ",\"velocity_wind\":[%.3f,%.3f,%.3f]",
                                     s.windNed[0], s.windNed[1], s.windNed[2]);
            }
            if (!std::isnan(s.batteryV)) {
                len += std::snprintf(json + len, sizeof json - len,
                                     ",\"battery\":{\"voltage\":%.2f,\"current\":%.2f}",
                                     s.batteryV, std::isnan(s.batteryA) ? 0.0 : s.batteryA);
            }
            len += std::snprintf(json + len, sizeof json - len, "}\n");

            ::sendto(sock_, json, len, 0,
                     reinterpret_cast<const sockaddr*>(&peer_), sizeof peer_);
        }

        [[nodiscard]] bool connected() const { return connected_; }
        [[nodiscard]] std::uint32_t lastFrameCount() const { return lastFrameCount_; }

        /// "a.b.c.d:port" of the SITL instance, or "" before first contact.
        [[nodiscard]] std::string peer() const {
            if (!connected_) return {};
            char ip[INET_ADDRSTRLEN] = {};
            inet_ntop(AF_INET, &peer_.sin_addr, ip, sizeof ip);
            return std::string(ip) + ":" + std::to_string(ntohs(peer_.sin_port));
        }

        /// EMA of the achieved frame rate over wall time, for HUD display.
        [[nodiscard]] double achievedRateHz() const { return rateEma_; }

        /// Decode a raw datagram. Public + static so tests hit it directly.
        static bool decode(const std::uint8_t* buf, int n, ServoInput& out) {
            if (n < 8) return false;
            std::uint16_t magic;
            std::memcpy(&magic, buf, 2);
            int channels;
            if (magic == 18458 && n == 8 + 16 * 2) {
                channels = 16;
            } else if (magic == 29569 && n == 8 + 32 * 2) {
                channels = 32;
            } else {
                return false;
            }
            std::memcpy(&out.frameRate, buf + 2, 2);
            std::memcpy(&out.frameCount, buf + 4, 4);
            std::memset(out.pwm, 0, sizeof out.pwm);
            std::memcpy(out.pwm, buf + 8, channels * 2);
            out.channels = channels;
            return out.frameRate > 0;
        }

        /// Encode a servo packet (the selftest's fake SITL uses this).
        static int encode(const ServoInput& in, std::uint8_t (&buf)[128]) {
            const std::uint16_t magic = in.channels == 32 ? 29569 : 18458;
            const int channels = in.channels == 32 ? 32 : 16;
            std::memcpy(buf, &magic, 2);
            std::memcpy(buf + 2, &in.frameRate, 2);
            std::memcpy(buf + 4, &in.frameCount, 4);
            std::memcpy(buf + 8, in.pwm, channels * 2);
            return 8 + channels * 2;
        }

    private:
        void trackRate() {
            const auto now = std::chrono::steady_clock::now();
            if (haveFrame_) {
                const double dt = std::chrono::duration<double>(now - lastFrameWall_).count();
                if (dt > 0 && dt < 1.0) {
                    const double inst = 1.0 / dt;
                    rateEma_ = rateEma_ == 0 ? inst : rateEma_ + 0.05 * (inst - rateEma_);
                }
            }
            lastFrameWall_ = now;
        }

        socket_t sock_ = invalidSocket;
        std::uint16_t port_ = 0;
        sockaddr_in peer_{};
        bool connected_ = false;
        bool haveFrame_ = false;
        bool warnedBadPacket_ = false;
        std::uint32_t lastFrameCount_ = 0;
        std::chrono::steady_clock::time_point lastFrameWall_{};
        double rateEma_ = 0;
    };

}// namespace sitl

#endif// THREEPP_EXAMPLE_SITL_BRIDGE_HPP
