// Sensor authoring: the userData round-trip the editor's inspector writes.
//
// PhysX-free on purpose — this is the half CI can run, and it is the half that
// carries the format contract:
//
//   * every key is emitted on every write, whatever the type. A user who flips
//     the type combo to look at the depth-camera fields and flips it back must
//     find their beam pattern where they left it. The same lesson PhysicsConfig's
//     soft-body parameters carry, and the reason `encode()` has no branches.
//   * an unknown key is ignored rather than fatal, so a document written by a
//     newer editor still loads.
//   * `enabled == false` leaves no trace in the saved file.
//
// The live-sensor behaviour (registration, rate gating, recording) needs the
// PhysX SDK and lives in EditorSensorPlay_test.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/core/Object3D.hpp"

#include <set>
#include <string>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // The keys present in an encoded string, in no particular order.
    std::set<std::string> keysOf(const std::string& text) {

        std::set<std::string> keys;
        std::size_t start = 0;
        while (start <= text.size()) {
            const auto end = text.find(';', start);
            const auto token = text.substr(start, (end == std::string::npos ? text.size() : end) - start);
            const auto eq = token.find('=');
            if (eq != std::string::npos) keys.insert(token.substr(0, eq));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return keys;
    }

    // A config with every field moved off its default, so a round-trip that
    // drops a field is a failure rather than a coincidence.
    SensorConfig loaded() {

        SensorConfig config;
        config.enabled = true;
        config.type = SensorConfig::Type::Lidar;
        config.rateHz = 12.5f;
        config.seed = 4242;
        config.gyroNoiseDensity = 0.0011f;
        config.gyroRandomWalk = 2.5e-5f;
        config.accelNoiseDensity = 0.031f;
        config.accelRandomWalk = 1.5e-3f;
        config.nearPlane = 0.25f;
        config.farPlane = 47.5f;
        config.rangeStddev = 0.013f;
        config.rangeStddevPerMetre = 0.002f;
        config.rangeBias = -0.004f;
        config.fovY = 72.5f;
        config.width = 320;
        config.height = 240;
        config.beams = SensorConfig::Beams::OS0_128;
        config.faceSize = 256;
        config.encoderResolution = 0.001f;
        config.contactForceThreshold = 2.5f;
        config.joint = "shoulder_pan_joint";
        return config;
    }

}// namespace


TEST_CASE("A sensor config survives an encode/decode round trip") {

    const auto before = loaded();
    const auto after = SensorConfig::decode(before.encode());

    REQUIRE(after.has_value());
    // decode() sets enabled: carrying the entry IS being enabled.
    CHECK(after->enabled);
    CHECK(after->type == before.type);
    CHECK_THAT(after->rateHz, WithinAbs(before.rateHz, 1e-5));
    CHECK(after->seed == before.seed);
    CHECK_THAT(after->gyroNoiseDensity, WithinAbs(before.gyroNoiseDensity, 1e-7));
    CHECK_THAT(after->gyroRandomWalk, WithinAbs(before.gyroRandomWalk, 1e-9));
    CHECK_THAT(after->accelNoiseDensity, WithinAbs(before.accelNoiseDensity, 1e-7));
    CHECK_THAT(after->accelRandomWalk, WithinAbs(before.accelRandomWalk, 1e-8));
    CHECK_THAT(after->nearPlane, WithinAbs(before.nearPlane, 1e-6));
    CHECK_THAT(after->farPlane, WithinAbs(before.farPlane, 1e-5));
    CHECK_THAT(after->rangeStddev, WithinAbs(before.rangeStddev, 1e-7));
    CHECK_THAT(after->rangeStddevPerMetre, WithinAbs(before.rangeStddevPerMetre, 1e-8));
    CHECK_THAT(after->rangeBias, WithinAbs(before.rangeBias, 1e-8));
    CHECK_THAT(after->fovY, WithinAbs(before.fovY, 1e-5));
    CHECK(after->width == before.width);
    CHECK(after->height == before.height);
    CHECK(after->beams == before.beams);
    CHECK(after->faceSize == before.faceSize);
    CHECK_THAT(after->encoderResolution, WithinAbs(before.encoderResolution, 1e-9));
    CHECK_THAT(after->contactForceThreshold, WithinAbs(before.contactForceThreshold, 1e-6));
    CHECK(after->joint == before.joint);

    // Byte-identical on re-encode: the format has to be a fixed point, or a
    // save/load cycle dirties every document it touches.
    CHECK(after->encode() == before.encode());
}

TEST_CASE("Every type emits every key, so a type flip is not a data loss") {

    // The key set is what the claim is about, so pin it explicitly rather than
    // comparing two encodes to each other (which would pass if both were empty).
    const std::set<std::string> expected{
            "type", "rate", "seed",
            "gyrodensity", "gyrowalk", "acceldensity", "accelwalk",
            "near", "far", "rangestddev", "rangepermetre", "rangebias",
            "fov", "width", "height",
            "beams", "facesize",
            "joint", "encoderres", "contactthreshold"};

    for (const auto type : {SensorConfig::Type::Imu, SensorConfig::Type::Depth,
                            SensorConfig::Type::Lidar, SensorConfig::Type::Encoder,
                            SensorConfig::Type::Contact, SensorConfig::Type::ForceTorque,
                            SensorConfig::Type::Camera}) {
        auto config = loaded();
        config.type = type;
        CHECK(keysOf(config.encode()) == expected);
    }
}

TEST_CASE("Flipping the type and back keeps the other type's settings") {

    Object3D object;

    auto lidar = loaded();
    lidar.beams = SensorConfig::Beams::HDL32E;
    lidar.faceSize = 512;
    lidar.write(object);

    // The inspector's type combo: read, change one field, write back.
    auto asImu = SensorConfig::read(object).value();
    asImu.type = SensorConfig::Type::Imu;
    asImu.gyroNoiseDensity = 0.09f;
    asImu.write(object);

    auto back = SensorConfig::read(object).value();
    back.type = SensorConfig::Type::Lidar;
    back.write(object);

    const auto final = SensorConfig::read(object).value();
    CHECK(final.type == SensorConfig::Type::Lidar);
    CHECK(final.beams == SensorConfig::Beams::HDL32E);
    CHECK(final.faceSize == 512);
    // And the IMU edit made while the lidar fields were hidden is still there.
    CHECK_THAT(final.gyroNoiseDensity, WithinAbs(0.09f, 1e-7));
}

TEST_CASE("An object with no sensor entry reads as no sensor") {

    Object3D object;
    CHECK_FALSE(SensorConfig::read(object).has_value());

    // A non-string userData entry under the same key is not a config either.
    object.userData[SensorConfig::userDataKey] = 42;
    CHECK_FALSE(SensorConfig::read(object).has_value());
}

TEST_CASE("Disabling a sensor removes the entry from the object") {

    Object3D object;
    auto config = loaded();
    config.write(object);
    REQUIRE(object.userData.count(SensorConfig::userDataKey) == 1);

    config.enabled = false;
    config.write(object);
    CHECK(object.userData.count(SensorConfig::userDataKey) == 0);
    CHECK_FALSE(SensorConfig::read(object).has_value());
}

TEST_CASE("Unknown keys and junk are ignored, not fatal") {

    // A future editor's key, an empty token, and a valueless one.
    const auto config = SensorConfig::decode("type=lidar;;whatsthis=7;rate=25;garbage;facesize=64");
    REQUIRE(config.has_value());
    CHECK(config->type == SensorConfig::Type::Lidar);
    CHECK_THAT(config->rateHz, WithinAbs(25.f, 1e-6));
    CHECK(config->faceSize == 64);
    // An unreadable value keeps the default rather than becoming zero.
    const auto bad = SensorConfig::decode("type=lidar;rate=abc");
    REQUIRE(bad.has_value());
    CHECK_THAT(bad->rateHz, WithinAbs(SensorConfig{}.rateHz, 1e-6));

    // An empty string is no config at all (that is how erase() reads).
    CHECK_FALSE(SensorConfig::decode("").has_value());
}

TEST_CASE("An unrecognised type token falls back rather than corrupting") {

    const auto config = SensorConfig::decode("type=neutrino;rate=10");
    REQUIRE(config.has_value());
    CHECK(config->type == SensorConfig{}.type);
}

TEST_CASE("The joint key carries a URDF joint name and an empty one round-trips") {

    // A chosen joint rides verbatim, underscores and all.
    SensorConfig chosen;
    chosen.enabled = true;
    chosen.type = SensorConfig::Type::Encoder;
    chosen.joint = "arm_left_1_joint";
    const auto back = SensorConfig::decode(chosen.encode());
    REQUIRE(back.has_value());
    CHECK(back->joint == "arm_left_1_joint");

    // The default is empty, which still encodes (joint=;) and decodes to empty
    // rather than crashing on a valueless token.
    SensorConfig unset;
    unset.enabled = true;
    unset.type = SensorConfig::Type::ForceTorque;
    const auto emptyBack = SensorConfig::decode(unset.encode());
    REQUIRE(emptyBack.has_value());
    CHECK(emptyBack->joint.empty());
}

TEST_CASE("Sub-streams of one authored seed are distinct and reproducible") {

    SensorConfig a;
    a.seed = 7;
    SensorConfig b;
    b.seed = 7;

    // Same authored seed -> same streams (a recorded run replays).
    CHECK(a.streamSeed(0) == b.streamSeed(0));
    CHECK(a.streamSeed(2) == b.streamSeed(2));
    // Different channels of one sensor must not share a stream, or the gyro and
    // the accelerometer would draw correlated noise.
    CHECK(a.streamSeed(0) != a.streamSeed(1));
    CHECK(a.streamSeed(1) != a.streamSeed(2));
    // And seed 0 is a usable stream, not a degenerate one.
    SensorConfig zero;
    zero.seed = 0;
    CHECK(zero.streamSeed(0) != 0u);
    CHECK(zero.streamSeed(0) != a.streamSeed(0));
}

TEST_CASE("Vision and proprioceptive types are partitioned") {

    CHECK(SensorConfig::isVision(SensorConfig::Type::Depth));
    CHECK(SensorConfig::isVision(SensorConfig::Type::Lidar));
    CHECK(SensorConfig::isVision(SensorConfig::Type::Camera));
    CHECK(SensorConfig::isProprioceptive(SensorConfig::Type::Imu));
    CHECK(SensorConfig::isProprioceptive(SensorConfig::Type::Contact));
    CHECK(SensorConfig::isProprioceptive(SensorConfig::Type::Encoder));
    CHECK(SensorConfig::isProprioceptive(SensorConfig::Type::ForceTorque));
}

TEST_CASE("A camera is vision but not ranging") {

    // The distinction the range-noise fields hang off: a Camera goes through
    // the renderer and the frame loop like the other two, but there is no
    // distance in a colour pixel for a sigma in metres to corrupt. Fold it into
    // isRanging and the inspector grows noise controls that do nothing, while
    // the point-cloud overlay starts asking a picture for its points.
    CHECK(SensorConfig::isRanging(SensorConfig::Type::Depth));
    CHECK(SensorConfig::isRanging(SensorConfig::Type::Lidar));
    CHECK_FALSE(SensorConfig::isRanging(SensorConfig::Type::Camera));
    CHECK_FALSE(SensorConfig::isRanging(SensorConfig::Type::Imu));
}

TEST_CASE("A camera survives the userData round trip") {

    SensorConfig camera;
    camera.enabled = true;
    camera.type = SensorConfig::Type::Camera;
    camera.rateHz = 15.f;
    camera.fovY = 42.5f;
    camera.width = 640;
    camera.height = 480;
    camera.nearPlane = 0.02f;
    camera.farPlane = 12.f;

    Object3D object;
    camera.write(object);

    const auto read = SensorConfig::read(object);
    REQUIRE(read.has_value());
    // The type token, not the enum's ordinal: a document written today has to
    // survive a Type appended tomorrow.
    CHECK(read->type == SensorConfig::Type::Camera);
    CHECK(read->width == 640);
    CHECK(read->height == 480);
    CHECK_THAT(read->fovY, WithinAbs(42.5, 1e-4));
    CHECK_THAT(read->rateHz, WithinAbs(15.0, 1e-4));
}
