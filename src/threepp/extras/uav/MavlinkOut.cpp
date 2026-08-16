// MAVLink v2 framing + the UDP socket that carries it. Platform socket headers
// stay in this translation unit — the public header must not leak winsock (or
// its near/far macro fallout) into consumers.

#include "threepp/extras/uav/MavlinkOut.hpp"

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

        constexpr std::uint8_t magicV2 = 0xFD;

        // CRC_EXTRA is a hash of the message's field signature; it makes a
        // wrong-dialect decoder reject the frame instead of misreading it.
        constexpr std::uint8_t crcExtraHeartbeat = 50;
        constexpr std::uint8_t crcExtraObstacleDistance = 23;

        constexpr std::uint32_t msgidHeartbeat = 0;
        constexpr std::uint32_t msgidObstacleDistance = 330;

        void put16(std::vector<std::uint8_t>& v, std::uint16_t x) {
            v.push_back(static_cast<std::uint8_t>(x & 0xff));
            v.push_back(static_cast<std::uint8_t>(x >> 8));
        }

        void put32(std::vector<std::uint8_t>& v, std::uint32_t x) {
            for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
        }

        void put64(std::vector<std::uint8_t>& v, std::uint64_t x) {
            for (int i = 0; i < 8; ++i) v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
        }

        void putFloat(std::vector<std::uint8_t>& v, float f) {
            std::uint32_t bits;
            std::memcpy(&bits, &f, 4);
            put32(v, bits);
        }

        /// Wrap a payload (already in wire field order) in a v2 frame. Named
        /// frameV2, not frame — threepp::uav::frame is the FrameConv namespace.
        std::vector<std::uint8_t> frameV2(std::uint8_t seq, std::uint8_t sysid, std::uint8_t compid,
                                          std::uint32_t msgid, std::vector<std::uint8_t> payload,
                                          std::uint8_t crcExtra) {
            // v2 TRAILING-ZERO TRUNCATION: zero bytes off the end are implied,
            // never transmitted. The CRC covers the TRUNCATED payload, so this
            // is not a transport-layer nicety we may skip — get the length
            // wrong and every frame fails the receiver's checksum.
            std::size_t len = payload.size();
            while (len > 1 && payload[len - 1] == 0) --len;

            std::vector<std::uint8_t> out;
            out.reserve(12 + len);
            out.push_back(magicV2);
            out.push_back(static_cast<std::uint8_t>(len));
            out.push_back(0);// incompat_flags: 0 = unsigned frame
            out.push_back(0);// compat_flags
            out.push_back(seq);
            out.push_back(sysid);
            out.push_back(compid);
            out.push_back(static_cast<std::uint8_t>(msgid & 0xff));
            out.push_back(static_cast<std::uint8_t>((msgid >> 8) & 0xff));
            out.push_back(static_cast<std::uint8_t>((msgid >> 16) & 0xff));
            out.insert(out.end(), payload.begin(),
                       payload.begin() + static_cast<std::ptrdiff_t>(len));

            // CRC runs from payload_len onward — the 0xFD magic is NOT covered.
            std::uint16_t crc = 0xFFFF;
            for (std::size_t i = 1; i < out.size(); ++i) mavlink::crcAccumulate(out[i], crc);
            mavlink::crcAccumulate(crcExtra, crc);
            put16(out, crc);
            return out;
        }

    }// namespace

    namespace mavlink {

        void crcAccumulate(std::uint8_t byte, std::uint16_t& crc) {
            std::uint8_t tmp = byte ^ static_cast<std::uint8_t>(crc & 0xff);
            tmp ^= static_cast<std::uint8_t>(tmp << 4);
            crc = static_cast<std::uint16_t>((crc >> 8) ^ (static_cast<std::uint16_t>(tmp) << 8) ^
                                             (static_cast<std::uint16_t>(tmp) << 3) ^
                                             (static_cast<std::uint16_t>(tmp) >> 4));
        }

        std::vector<std::uint8_t> packHeartbeat(std::uint8_t seq, std::uint8_t sysid,
                                                std::uint8_t compid) {
            // Wire order is by descending field size, NOT declaration order.
            std::vector<std::uint8_t> p;
            p.reserve(9);
            put32(p, 0);   // custom_mode
            p.push_back(18);// type: MAV_TYPE_ONBOARD_CONTROLLER
            p.push_back(8); // autopilot: MAV_AUTOPILOT_INVALID (we are not one)
            p.push_back(0); // base_mode
            p.push_back(4); // system_status: MAV_STATE_ACTIVE
            p.push_back(3); // mavlink_version
            return frameV2(seq, sysid, compid, msgidHeartbeat, std::move(p), crcExtraHeartbeat);
        }

        std::vector<std::uint8_t> packObstacleDistance(
                std::uint8_t seq, std::uint8_t sysid, std::uint8_t compid,
                std::uint64_t timeUsec, const Distances& distancesCm,
                std::uint16_t minCm, std::uint16_t maxCm, std::uint8_t incrementDeg,
                float incrementF, float angleOffsetDeg, std::uint8_t frameId) {
            std::vector<std::uint8_t> p;
            p.reserve(167);
            put64(p, timeUsec);
            for (std::uint16_t d : distancesCm) put16(p, d);
            put16(p, minCm);
            put16(p, maxCm);
            p.push_back(0);// sensor_type: MAV_DISTANCE_SENSOR_LASER
            p.push_back(incrementDeg);
            // MAVLink v2 extension fields — appended after the v1 payload, and
            // the first thing trailing-zero truncation eats when unused.
            putFloat(p, incrementF);
            putFloat(p, angleOffsetDeg);
            p.push_back(frameId);
            return frameV2(seq, sysid, compid, msgidObstacleDistance, std::move(p),
                           crcExtraObstacleDistance);
        }

    }// namespace mavlink

    struct MavlinkOut::Impl {
        socket_t sock = invalidSocket;
        std::uint16_t port = 0;
        std::uint8_t sysid = 1;
        std::uint8_t compid = mavlink::componentIdObstacleAvoidance;
        std::uint8_t seq = 0;
        sockaddr_in peer{};
        bool havePeer = false;
        std::uint64_t sent = 0;

        void send(const std::vector<std::uint8_t>& bytes) {
            if (sock == invalidSocket || !havePeer) return;
            ::sendto(sock, reinterpret_cast<const char*>(bytes.data()),
                     static_cast<int>(bytes.size()), 0,
                     reinterpret_cast<const sockaddr*>(&peer), sizeof peer);
            ++sent;
        }
    };

    MavlinkOut::MavlinkOut(std::uint16_t port, std::uint8_t sysid, std::uint8_t compid)
        : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
        static const int wsaInit = [] {
            WSADATA data;
            return WSAStartup(MAKEWORD(2, 2), &data);
        }();
        (void) wsaInit;
#endif
        impl_->sysid = sysid;
        impl_->compid = compid;

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

    MavlinkOut::~MavlinkOut() {
        if (impl_->sock != invalidSocket) closesocket(impl_->sock);
    }

    bool MavlinkOut::valid() const {
        return impl_->sock != invalidSocket;
    }

    std::uint16_t MavlinkOut::port() const {
        return impl_->port;
    }

    int MavlinkOut::poll() {
        if (impl_->sock == invalidSocket) return 0;

        int got = 0;
        // Drain fully: an undrained UDP socket eventually drops, and the newest
        // datagram is the freshest return address.
        for (;;) {
            std::uint8_t buf[1024];
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
            if (n <= 0) break;
            impl_->peer = from;
            impl_->havePeer = true;
            ++got;
        }
        return got;
    }

    bool MavlinkOut::hasPeer() const {
        return impl_->havePeer;
    }

    std::string MavlinkOut::peer() const {
        if (!impl_->havePeer) return {};
        char ip[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &impl_->peer.sin_addr, ip, sizeof ip);
        return std::string(ip) + ":" + std::to_string(ntohs(impl_->peer.sin_port));
    }

    void MavlinkOut::sendHeartbeat() {
        if (!impl_->havePeer) return;
        impl_->send(mavlink::packHeartbeat(impl_->seq++, impl_->sysid, impl_->compid));
    }

    void MavlinkOut::sendObstacleDistance(std::uint64_t timeUsec,
                                          const mavlink::Distances& distancesCm,
                                          std::uint16_t minCm, std::uint16_t maxCm,
                                          std::uint8_t incrementDeg, float angleOffsetDeg,
                                          std::uint8_t frameId) {
        if (!impl_->havePeer) return;
        // increment_f stays 0: the integer `increment` is exact at 5 deg, and
        // ArduPilot prefers the float only when it is non-zero.
        impl_->send(mavlink::packObstacleDistance(impl_->seq++, impl_->sysid, impl_->compid,
                                                  timeUsec, distancesCm, minCm, maxCm,
                                                  incrementDeg, 0.f, angleOffsetDeg, frameId));
    }

    std::uint64_t MavlinkOut::sentCount() const {
        return impl_->sent;
    }

}// namespace threepp::uav
