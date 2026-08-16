// MAVLink v2 framing against pymavlink 2.4.49 golden bytes, plus the sector
// geometry that fills the message. The goldens are the point: a hand-rolled
// framer is only worth having if a byte-for-byte reference pins it, and the
// two failure modes that survive code review (trailing-zero truncation, CRC
// coverage starting AFTER the magic) are exactly the ones a golden catches.

#include "threepp/extras/uav/MavlinkOut.hpp"
#include "threepp/extras/uav/ProximityScan.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::uav;

namespace {

    std::vector<std::uint8_t> fromHex(const std::string& hex) {
        std::vector<std::uint8_t> out;
        out.reserve(hex.size() / 2);
        for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
            out.push_back(static_cast<std::uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
        }
        return out;
    }

}// namespace

TEST_CASE("HEARTBEAT matches the pymavlink golden") {
    const auto bytes = mavlink::packHeartbeat(7, 1, mavlink::componentIdObstacleAvoidance);
    CHECK(bytes == fromHex("fd0900000701c4000000000000001208000403b728"));
}

TEST_CASE("OBSTACLE_DISTANCE matches the pymavlink golden") {
    mavlink::Distances d;
    d.fill(5001);// clear, per ProximityScan
    d[0] = 123;
    d[1] = 65535;// UNKNOWN is legal on the wire, we just never produce it
    d[35] = 4999;

    const auto bytes = mavlink::packObstacleDistance(
            8, 1, mavlink::componentIdObstacleAvoidance, 1234567, d, 30, 5000, 5,
            0.f, 0.f, mavlink::frameBodyFrd);

    // frame=12 is the last payload byte and non-zero, so nothing truncates:
    // full 167-byte payload + 12 bytes of frame.
    CHECK(bytes.size() == 179);
    CHECK(bytes == fromHex(
                           "fda700000801c44a010087d61200000000007b00ffff8913891389138913891389138913891389138913891389"
                           "138913891389138913891389138913891389138913891389138913891389138913891389138913891389138713"
                           "891389138913891389138913891389138913891389138913891389138913891389138913891389138913891389"
                           "1389138913891389138913891389138913891389138913891389131e008813000500000000000000000cb0b8"));
}

TEST_CASE("trailing zeros are truncated off the payload") {
    mavlink::Distances d;
    d.fill(0);
    d[0] = 250;

    // increment_f, angle_offset and frame all zero: 9 bytes vanish off the
    // tail (167 -> 158), and the CRC must be computed over the SHORT payload.
    const auto bytes = mavlink::packObstacleDistance(9, 1, mavlink::componentIdObstacleAvoidance,
                                                     1, d, 30, 5000, 5, 0.f, 0.f, 0);
    CHECK(bytes[1] == 158);
    CHECK(bytes.size() == 170);
    CHECK(bytes == fromHex(
                           "fd9e00000901c44a01000100000000000000fa0000000000000000000000000000000000000000000000000000"
                           "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                           "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                           "0000000000000000000000000000000000000000000000000000001e0088130005241a"));
}

TEST_CASE("sectors map clockwise from the nose, in the yaw frame") {
    // A wall due East (+X) at 10 m and nothing else: only rays within 3 deg of
    // due East report.
    const RayCaster east = [](const Vector3&, const Vector3& dir, float) {
        const float az = std::atan2(dir.x, -dir.z);// clockwise from North
        const float eastAz = 1.5707963f;
        return std::abs(az - eastAz) <= 0.05236f ? 10.f : -1.f;
    };

    ProximityScan scan;
    const auto& d = scan.scan(east, Vector3(0.f, 5.f, 0.f), Vector3(0.f, 0.f, -1.f));

    // Sector 18 spans 90..95 deg clockwise from the nose — due East.
    CHECK(d[18] == 1000);
    CHECK(d[54] == scan.clearValue());// due West, and NOT 65535
    CHECK(scan.clearValue() == 5001);

    // Pitching the nose down does not tilt the fan: the rays stay horizontal
    // in the world frame, so a diving quad still scans the horizon, not dirt.
    ProximityScan pitched;
    Vector3 fwd(0.f, -0.5f, -1.f);
    fwd.normalize();
    const auto& p = pitched.scan(east, Vector3(0.f, 5.f, 0.f), fwd);
    CHECK(p == d);
}

TEST_CASE("a fed return lands in the sector its bearing falls in") {
    ProximityScan scan;
    scan.beginFrame(Vector3(0.f, 5.f, 0.f), Vector3(0.f, 0.f, -1.f));// nose North
    // Due East (+X), 10 m out, level with the sensor — the same wall the ray
    // fan above reports, arriving as a point instead of a cast.
    scan.feed(Vector3(10.f, 5.f, 0.f));

    CHECK(scan.distances()[18] == 1000);
    CHECK(scan.distances()[54] == scan.clearValue());// due West stays measured-clear
}

TEST_CASE("returns outside the elevation slab are ignored") {
    ProximityScan scan;
    scan.beginFrame(Vector3(0.f, 5.f, 0.f), Vector3(0.f, 0.f, -1.f));

    // The same East point, 5 m BELOW the sensor: that is the ground a real
    // scanner sees on every downward ring, and a wall is what it must not
    // become.
    scan.feed(Vector3(10.f, 0.f, 0.f));
    CHECK(scan.distances()[18] == scan.clearValue());

    // The canopy arching overhead is out on the other side.
    scan.feed(Vector3(10.f, 12.f, 0.f));
    CHECK(scan.distances()[18] == scan.clearValue());

    // Inside the slab it counts, so the two above were rejected on height and
    // not on some sector-arithmetic accident.
    scan.feed(Vector3(10.f, 6.5f, 0.f));
    CHECK(scan.distances()[18] == 1000);
}

TEST_CASE("the nearest return wins its sector") {
    ProximityScan scan;
    scan.beginFrame(Vector3(0.f, 5.f, 0.f), Vector3(0.f, 0.f, -1.f));

    scan.feed(Vector3(10.f, 5.f, 0.f));
    scan.feed(Vector3(4.f, 5.f, 0.f));
    scan.feed(Vector3(12.f, 5.f, 0.f));// arrives last, must not win
    CHECK(scan.distances()[18] == 400);

    // HORIZONTAL distance: 4 m out and 2 m up is a thing 4 m ahead, not 4.47.
    ProximityScan tilted;
    tilted.beginFrame(Vector3(0.f, 5.f, 0.f), Vector3(0.f, 0.f, -1.f));
    tilted.feed(Vector3(4.f, 7.f, 0.f));
    CHECK(tilted.distances()[18] == 400);
}
