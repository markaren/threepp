// The MAVLink v2 half of the obstacle-avoidance feed: exactly two messages,
// hand-rolled. Vendoring pymavlink's generated headers would drag ~200 message
// definitions and a build-time Python codegen step into the tree for the sake
// of HEARTBEAT and OBSTACLE_DISTANCE, so the framer lives here instead. The
// wire format is pinned by golden byte strings in MavlinkOut_test.
//
// Talks to an ArduPilot SITL started with
//     --serial2=udpclient:127.0.0.1:<port>
// which means SITL is the one dialling out: it sends TO our bound port, and we
// only learn where to answer once a datagram arrives. Hence poll() and
// hasPeer() — sending before first contact has nowhere to go.
//
// The socket lives behind an Impl in MavlinkOut.cpp so this header pulls in no
// platform headers (winsock must not leak into consumers), same as SitlBridge.

#ifndef THREEPP_EXTRAS_UAV_MAVLINKOUT_HPP
#define THREEPP_EXTRAS_UAV_MAVLINKOUT_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace threepp::uav {

    namespace mavlink {

        /// OBSTACLE_DISTANCE carries a fixed 72-slot ring, always.
        inline constexpr std::size_t sectorCount = 72;
        using Distances = std::array<std::uint16_t, sectorCount>;

        /// MAV_COMP_ID_OBSTACLE_AVOIDANCE — what a proximity source announces
        /// itself as, and what ArduPilot's AP_Proximity_MAV expects to see.
        inline constexpr std::uint8_t componentIdObstacleAvoidance = 196;

        /// MAV_FRAME_BODY_FRD: sector 0 at the nose, advancing clockwise.
        inline constexpr std::uint8_t frameBodyFrd = 12;

        /// X.25 CRC as MAVLink defines it (seed 0xFFFF, one byte at a time).
        /// Exposed for tests; framing below uses it internally.
        void crcAccumulate(std::uint8_t byte, std::uint16_t& crc);

        /// HEARTBEAT (msgid 0): type ONBOARD_CONTROLLER, autopilot INVALID,
        /// status ACTIVE. ArduPilot ignores proximity data from a component it
        /// has never heard a heartbeat from, so this is not optional garnish.
        std::vector<std::uint8_t> packHeartbeat(std::uint8_t seq, std::uint8_t sysid,
                                                std::uint8_t compid);

        /// OBSTACLE_DISTANCE (msgid 330). `distancesCm` is one entry per
        /// `increment` degrees starting at `angleOffsetDeg` from the frame's
        /// forward; UINT16_MAX means UNKNOWN (see ProximityScan for why we do
        /// not send that for measured free space).
        std::vector<std::uint8_t> packObstacleDistance(
                std::uint8_t seq, std::uint8_t sysid, std::uint8_t compid,
                std::uint64_t timeUsec, const Distances& distancesCm,
                std::uint16_t minCm, std::uint16_t maxCm, std::uint8_t incrementDeg,
                float incrementF, float angleOffsetDeg, std::uint8_t frame);

    }// namespace mavlink

    /// UDP endpoint that speaks the two messages above at a peer it discovers.
    class MavlinkOut {

    public:
        /// Binds `port` on all interfaces (0 picks an ephemeral one — selftests
        /// use that to stay clear of a real SITL session). Handles winsock
        /// startup on Windows.
        explicit MavlinkOut(std::uint16_t port = 14560, std::uint8_t sysid = 1,
                            std::uint8_t compid = mavlink::componentIdObstacleAvoidance);

        ~MavlinkOut();

        MavlinkOut(const MavlinkOut&) = delete;
        MavlinkOut& operator=(const MavlinkOut&) = delete;

        /// False if the constructor failed to bind (port in use, no winsock).
        [[nodiscard]] bool valid() const;

        /// The actually-bound port (differs from the request only for port 0).
        [[nodiscard]] std::uint16_t port() const;

        /// Drain the socket, remembering the last sender as the peer. The
        /// content is discarded — we are a one-way source; what SITL sends us
        /// (its own heartbeats, param traffic) is only useful as a return
        /// address. Returns the number of datagrams consumed.
        int poll();

        [[nodiscard]] bool hasPeer() const;

        /// "a.b.c.d:port" of the discovered peer, or "" before first contact.
        [[nodiscard]] std::string peer() const;

        /// No-ops until a peer is known. Both share one sequence counter, per
        /// MAVLink convention (seq is per-link, not per-message).
        void sendHeartbeat();
        void sendObstacleDistance(std::uint64_t timeUsec, const mavlink::Distances& distancesCm,
                                  std::uint16_t minCm, std::uint16_t maxCm,
                                  std::uint8_t incrementDeg, float angleOffsetDeg,
                                  std::uint8_t frame);

        [[nodiscard]] std::uint64_t sentCount() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

}// namespace threepp::uav

#endif// THREEPP_EXTRAS_UAV_MAVLINKOUT_HPP
