// Socket half of the ArduPilot SITL JSON bridge. Platform socket headers stay
// in this translation unit — the public header must not leak winsock (or its
// near/far macro fallout) into consumers.

#include "threepp/extras/uav/SitlBridge.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstring>

namespace threepp::uav {

    namespace {

#ifdef _WIN32
        using socket_t = SOCKET;
        constexpr socket_t invalidSocket = INVALID_SOCKET;
#else
        using socket_t = int;
        constexpr socket_t invalidSocket = -1;
        void closesocket(socket_t s) { ::close(s); }
#endif

    }// namespace

    struct SitlBridge::Impl {
        socket_t sock = invalidSocket;
        std::uint16_t port = 0;
        sockaddr_in peer{};
        bool connected = false;
        bool haveFrame = false;
        bool warnedBadPacket = false;
        std::uint32_t lastFrameCount = 0;
        std::chrono::steady_clock::time_point lastFrameWall{};
        double rateEma = 0;

        void trackRate() {
            const auto now = std::chrono::steady_clock::now();
            if (haveFrame) {
                const double dt = std::chrono::duration<double>(now - lastFrameWall).count();
                if (dt > 0 && dt < 1.0) {
                    const double inst = 1.0 / dt;
                    rateEma = rateEma == 0 ? inst : rateEma + 0.05 * (inst - rateEma);
                }
            }
            lastFrameWall = now;
        }
    };

    SitlBridge::SitlBridge(std::uint16_t port): impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
        static const int wsaInit = [] {
            WSADATA data;
            return WSAStartup(MAKEWORD(2, 2), &data);
        }();
        (void) wsaInit;
#endif
        impl_->sock = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (impl_->sock == invalidSocket) return;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        if (::bind(impl_->sock, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
            closesocket(impl_->sock);
            impl_->sock = invalidSocket;
            return;
        }
#ifdef _WIN32
        u_long nonBlocking = 1;
        ioctlsocket(impl_->sock, FIONBIO, &nonBlocking);
#else
        fcntl(impl_->sock, F_SETFL, fcntl(impl_->sock, F_GETFL, 0) | O_NONBLOCK);
#endif
        sockaddr_in bound{};
#ifdef _WIN32
        int len = sizeof bound;
#else
        socklen_t len = sizeof bound;
#endif
        if (::getsockname(impl_->sock, reinterpret_cast<sockaddr*>(&bound), &len) == 0) {
            impl_->port = ntohs(bound.sin_port);
        }
    }

    SitlBridge::~SitlBridge() {
        if (impl_->sock != invalidSocket) closesocket(impl_->sock);
    }

    bool SitlBridge::valid() const {
        return impl_->sock != invalidSocket;
    }

    std::uint16_t SitlBridge::port() const {
        return impl_->port;
    }

    SitlBridge::Event SitlBridge::poll(ServoInput& out) {
        if (impl_->sock == invalidSocket) return Event::None;

        std::uint8_t buf[128];
        sockaddr_in from{};
#ifdef _WIN32
        int fromLen = sizeof from;
        const int n = ::recvfrom(impl_->sock, reinterpret_cast<char*>(buf), sizeof buf, 0,
                                 reinterpret_cast<sockaddr*>(&from), &fromLen);
#else
        socklen_t fromLen = sizeof from;
        const auto n = ::recvfrom(impl_->sock, buf, sizeof buf, 0,
                                  reinterpret_cast<sockaddr*>(&from), &fromLen);
#endif
        if (n <= 0) return Event::None;

        if (!decode(buf, static_cast<int>(n), out)) {
            if (!impl_->warnedBadPacket) {
                std::fprintf(stderr, "[sitl] dropping unrecognized %d-byte packet (bad magic/length)\n",
                             static_cast<int>(n));
                impl_->warnedBadPacket = true;
            }
            return Event::None;
        }

        const bool newPeer = from.sin_addr.s_addr != impl_->peer.sin_addr.s_addr ||
                             from.sin_port != impl_->peer.sin_port;
        const sockaddr_in prevPeer = impl_->peer;
        impl_->peer = from;
        impl_->connected = true;
        impl_->trackRate();

        // frame_count going backwards means SITL restarted; SITL also
        // re-sends the SAME frame every 10 s while unanswered, so equality
        // is a retry, not a restart.
        const bool restarted = (newPeer && impl_->haveFrame) ||
                               (impl_->haveFrame && out.frameCount < impl_->lastFrameCount);
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
                         impl_->lastFrameCount, out.frameCount);
        }
        impl_->haveFrame = true;
        impl_->lastFrameCount = out.frameCount;
        return restarted ? Event::Reset : Event::Frame;
    }

    void SitlBridge::sendState(const FdmState& s) {
        if (impl_->sock == invalidSocket || !impl_->connected) return;

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

        ::sendto(impl_->sock, json, len, 0,
                 reinterpret_cast<const sockaddr*>(&impl_->peer), sizeof impl_->peer);
    }

    bool SitlBridge::connected() const {
        return impl_->connected;
    }

    std::uint32_t SitlBridge::lastFrameCount() const {
        return impl_->lastFrameCount;
    }

    std::string SitlBridge::peer() const {
        if (!impl_->connected) return {};
        char ip[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &impl_->peer.sin_addr, ip, sizeof ip);
        return std::string(ip) + ":" + std::to_string(ntohs(impl_->peer.sin_port));
    }

    double SitlBridge::achievedRateHz() const {
        return impl_->rateEma;
    }

    bool SitlBridge::decode(const std::uint8_t* buf, int n, ServoInput& out) {
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

    int SitlBridge::encode(const ServoInput& in, std::uint8_t (&buf)[128]) {
        const std::uint16_t magic = in.channels == 32 ? 29569 : 18458;
        const int channels = in.channels == 32 ? 32 : 16;
        std::memcpy(buf, &magic, 2);
        std::memcpy(buf + 2, &in.frameRate, 2);
        std::memcpy(buf + 4, &in.frameCount, 4);
        std::memcpy(buf + 8, in.pwm, channels * 2);
        return 8 + channels * 2;
    }

}// namespace threepp::uav
