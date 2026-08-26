// The vision-sensor contract: seeded range noise, sim-clock timestamps.
//
// PhysX-free and renderer-free on purpose — everything here is the part of a
// depth/LIDAR scan that decides whether a generated dataset can be replayed,
// and it must run in CI where there is neither a GPU nor the PhysX SDK. The
// scan geometry itself is exercised by the examples; what is pinned here is the
// property the old `static std::mt19937 rng{std::random_device{}()}` could not
// offer: given a seed and a beam order, the same noise, every run, every
// machine, regardless of what the sensor next door is doing.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/VisionSensor.hpp"
#include "threepp/helpers/EventCameraSensor.hpp"

#include <cmath>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    // Concrete VisionSensor with the protected scan hooks opened up, standing in
    // for DepthSensor / LidarSensor / PathTracedLidarSensor — all three inherit
    // exactly this machinery and differ only in how they obtain a clean range.
    class TestVisionSensor: public VisionSensor {

    public:
        using VisionSensor::VisionSensor;
        using VisionSensor::applyRangeNoise;
        using VisionSensor::beginScan;

        // One "scan" of n beams at a constant clean range.
        std::vector<float> scan(int n, float range = 10.f) {
            beginScan();
            std::vector<float> out;
            out.reserve(n);
            for (int i = 0; i < n; ++i) out.push_back(applyRangeNoise(range));
            return out;
        }
    };

    struct Stats {
        double mean = 0.0;
        double stddev = 0.0;
    };

    Stats statsOf(const std::vector<float>& v) {
        Stats s;
        for (float x: v) s.mean += x;
        s.mean /= static_cast<double>(v.size());
        for (float x: v) s.stddev += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(s.stddev / static_cast<double>(v.size()));
        return s;
    }

}// namespace


// ---------------------------------------------------------------------------
// Range noise: the perfect sensor
// ---------------------------------------------------------------------------

TEST_CASE("a zero range-noise model is a bit-exact passthrough", "[sensors]") {

    auto node = Object3D::create();
    TestVisionSensor sensor(*node, RangeNoiseModel{});

    // Ground-truth captures depend on this: no rounding, no drift, no "almost".
    for (float r: {0.f, 0.001f, 1.f, 7.3125f, 99.9f, 1000.f}) {
        REQUIRE(sensor.applyRangeNoise(r) == r);
    }

    // And it must not consume the RNG either, or a capture taken with noise off
    // would silently shift the stream for whatever runs next.
    sensor.rangeNoise.stddev = 0.05f;
    TestVisionSensor fresh(*node, RangeNoiseModel{});
    fresh.rangeNoise.stddev = 0.05f;
    REQUIRE(sensor.applyRangeNoise(10.f) == fresh.applyRangeNoise(10.f));
}

// ---------------------------------------------------------------------------
// Range noise: determinism
// ---------------------------------------------------------------------------

TEST_CASE("the same seed reproduces the same scan", "[sensors]") {

    auto node = Object3D::create();
    const RangeNoiseModel model{/*stddev*/ 0.02f, /*perMetre*/ 0.f, /*bias*/ 0.f, /*seed*/ 12345};

    TestVisionSensor a(*node, model);
    TestVisionSensor b(*node, model);

    const auto first = a.scan(256);
    const auto second = b.scan(256);
    REQUIRE(first == second);// bit-for-bit, not merely close

    // A different seed must actually produce a different stream (a seed that is
    // silently ignored would pass every test above).
    RangeNoiseModel other = model;
    other.seed = 12346;
    TestVisionSensor c(*node, other);
    REQUIRE(c.scan(256) != first);
}

TEST_CASE("one sensor's noise stream is independent of another's", "[sensors]") {

    // THE regression this port exists for. The pre-port sensors drew from a
    // function-local `static` RNG, so a sensor's noise depended on how many
    // times every other sensor in the process had been scanned — two identical
    // rigs recorded in different orders produced different datasets.
    auto node = Object3D::create();
    const RangeNoiseModel model{0.02f, 0.f, 0.f, /*seed*/ 777};

    TestVisionSensor solo(*node, model);
    const auto reference = solo.scan(64);

    TestVisionSensor a(*node, model);
    TestVisionSensor b(*node, model);
    std::vector<float> interleaved;
    for (int i = 0; i < 64; ++i) {
        (void) a.applyRangeNoise(10.f);    // a neighbouring sensor, scanning away
        interleaved.push_back(b.applyRangeNoise(10.f));
    }

    REQUIRE(interleaved == reference);
}

TEST_CASE("range noise matches the model it was given", "[sensors]") {

    auto node = Object3D::create();

    SECTION("stddev and bias") {
        TestVisionSensor sensor(*node, RangeNoiseModel{/*stddev*/ 0.05f, /*perMetre*/ 0.f,
                                                       /*bias*/ 0.1f, /*seed*/ 9});
        const auto s = statsOf(sensor.scan(50000, 10.f));
        REQUIRE_THAT(s.mean, WithinAbs(10.1, 0.002));  // clean range + bias
        REQUIRE_THAT(s.stddev, WithinRel(0.05, 0.03));
    }

    SECTION("the range-proportional term adds in quadrature") {
        TestVisionSensor sensor(*node, RangeNoiseModel{/*stddev*/ 0.03f, /*perMetre*/ 0.001f,
                                                       /*bias*/ 0.f, /*seed*/ 9});
        const auto near = statsOf(sensor.scan(50000, 1.f));
        const auto far = statsOf(sensor.scan(50000, 100.f));

        REQUIRE_THAT(near.stddev, WithinRel(std::hypot(0.03, 0.001), 0.03));
        REQUIRE_THAT(far.stddev, WithinRel(std::hypot(0.03, 0.1), 0.03));
        // Noise grows with range — the whole point of the second term.
        REQUIRE(far.stddev > near.stddev * 2.0);
    }
}

TEST_CASE("tuning sigma keeps the stream, changing the seed restarts it", "[sensors]") {

    // An interactive noise slider writes stddev every frame. If that restarted
    // the RNG the noise would freeze into the geometry instead of shimmering,
    // so only a seed change re-seeds.
    auto node = Object3D::create();
    const RangeNoiseModel model{0.02f, 0.f, 0.f, /*seed*/ 4242};

    TestVisionSensor a(*node, model);
    TestVisionSensor b(*node, model);

    const auto firstA = a.scan(32);
    b.scan(32);

    a.rangeNoise.stddev = 0.02f;// same value re-assigned, as a slider would
    const auto secondA = a.scan(32);
    const auto secondB = b.scan(32);
    REQUIRE(secondA == secondB);       // the stream continued for both
    REQUIRE(secondA != firstA);        // and it is genuinely a new draw

    // A seed change takes effect on the next scan, and matches a sensor that
    // was constructed with that seed from the start.
    a.rangeNoise.seed = 555;
    const auto reseeded = a.scan(32);
    RangeNoiseModel other = model;
    other.seed = 555;
    TestVisionSensor c(*node, other);
    REQUIRE(reseeded == c.scan(32));

    // resetNoise() replays the episode from the top.
    c.resetNoise();
    REQUIRE(c.scan(32) == reseeded);
}

// ---------------------------------------------------------------------------
// Sim clock
// ---------------------------------------------------------------------------

TEST_CASE("scans are stamped with the sim clock", "[sensors]") {

    auto node = Object3D::create();
    TestVisionSensor sensor(*node, RangeNoiseModel{});

    // Undriven: an obviously-unstamped dataset rather than a plausible one.
    REQUIRE(sensor.simTime() == 0.0);
    REQUIRE(sensor.lastScanTime() == 0.0);

    sensor.advanceClock(0.25);
    sensor.advanceClock(0.25);
    sensor.scan(4);
    REQUIRE_THAT(sensor.lastScanTime(), WithinAbs(0.5, 1e-12));

    sensor.setSimTime(41.0);
    sensor.advanceClock(1.0);
    sensor.scan(4);
    REQUIRE_THAT(sensor.lastScanTime(), WithinAbs(42.0, 1e-12));

    // Driven by a step loop (what PhysxWorld::registerSensor does), the clock
    // tracks sim time with no help from the caller.
    for (int i = 1; i <= 10; ++i) sensor.tick(0.1, static_cast<double>(i) * 0.1);
    sensor.scan(4);
    REQUIRE_THAT(sensor.lastScanTime(), WithinAbs(1.0, 1e-9));
}

TEST_CASE("the clock advances on substeps that do not sample", "[sensors]") {

    // A rate-gated vision sensor is only "due" now and then, but a scan pulled
    // between two gate openings must still carry the CURRENT time, not the time
    // of the last gate opening.
    auto node = Object3D::create();
    TestVisionSensor sensor(*node, RangeNoiseModel{}, /*rateHz*/ 10.0);

    double t = 0.0;
    for (int i = 0; i < 25; ++i) {// 25 substeps of 4 ms = 0.1 s
        t += 0.004;
        sensor.tick(0.004, t);
    }
    REQUIRE_THAT(sensor.simTime(), WithinAbs(0.1, 1e-9));
}

TEST_CASE("the rate gate drives scanDue", "[sensors]") {

    auto node = Object3D::create();

    SECTION("an ungated sensor is always due") {
        TestVisionSensor sensor(*node, RangeNoiseModel{});
        REQUIRE(sensor.scanDue());
        sensor.scan(1);
        REQUIRE(sensor.scanDue());// scanning every frame is the default
    }

    SECTION("a gated sensor is due at its rate") {
        TestVisionSensor sensor(*node, RangeNoiseModel{}, /*rateHz*/ 10.0);
        REQUIRE_FALSE(sensor.scanDue());// nothing has driven it yet

        double t = 0.0;
        int due = 0;
        for (int i = 0; i < 500; ++i) {// 2 s at 250 Hz
            t += 0.004;
            sensor.tick(0.004, t);
            if (sensor.scanDue()) {
                ++due;
                sensor.scan(1);              // consumes the gate
                REQUIRE_FALSE(sensor.scanDue());
            }
        }
        REQUIRE(due == 20);// 10 Hz over 2 s
    }
}

// ---------------------------------------------------------------------------
// Event camera: same clock rule, no attachment node
// ---------------------------------------------------------------------------

TEST_CASE("event timestamps interpolate within the sim-clock interval", "[sensors]") {

    constexpr unsigned w = 8, h = 4;
    EventCameraSensor sensor(w, h);

    std::vector<unsigned char> dark(w * h * 3, 40);
    std::vector<unsigned char> bright(w * h * 3, 200);
    std::vector<EventCameraEvent> events;

    // First frame only establishes the per-pixel reference.
    sensor.ingestPixels(dark.data(), dark.size(), events);
    REQUIRE(events.empty());

    // A flat dark→bright step is ~1.61 log units, i.e. 10 thresholds' worth,
    // capped to maxEventsPerPixel = 5. Those 5 crossings happened at five
    // DIFFERENT points of the ramp, so they get five different stamps inside
    // (0, 0.5] — not one stamp at the frame boundary.
    sensor.advanceClock(0.5);
    sensor.ingestPixels(bright.data(), bright.size(), events);
    REQUIRE_FALSE(events.empty());
    for (const auto& e: events) {
        REQUIRE(e.polarity == 1);
        REQUIRE(e.timestamp > 0.f);
        REQUIRE(e.timestamp <= 0.5f);
    }
    // Sorted ascending, and genuinely spread rather than collapsed.
    for (std::size_t i = 1; i < events.size(); ++i) {
        REQUIRE(events[i].timestamp >= events[i - 1].timestamp);
    }
    REQUIRE(events.back().timestamp > events.front().timestamp);
    // Every pixel sees the identical step, so the distinct stamps are exactly
    // the per-pixel crossing count — the interpolation, not pixel-to-pixel
    // variation, is what spreads them.
    REQUIRE_THAT(events.front().timestamp,
                 WithinAbs(0.5f * 0.15f / std::log(200.f / 40.f), 1e-4f));

    // Jumping the clock forward stretches the ramp: the same five crossings
    // now land across (0.5, 9.0], still inside the interval they belong to.
    sensor.setSimTime(9.0);
    sensor.ingestPixels(dark.data(), dark.size(), events);
    REQUIRE_FALSE(events.empty());
    for (const auto& e: events) {
        REQUIRE(e.polarity == -1);
        REQUIRE(e.timestamp > 0.5f);
        REQUIRE(e.timestamp <= 9.0f);
    }

    // Replaying the same frames on the same clock reproduces the same stream —
    // which a wall clock could never do.
    EventCameraSensor replay(w, h);
    std::vector<EventCameraEvent> replayed;
    replay.ingestPixels(dark.data(), dark.size(), replayed);
    replay.advanceClock(0.5);
    replay.ingestPixels(bright.data(), bright.size(), replayed);

    std::vector<EventCameraEvent> original;
    EventCameraSensor again(w, h);
    again.ingestPixels(dark.data(), dark.size(), original);
    again.advanceClock(0.5);
    again.ingestPixels(bright.data(), bright.size(), original);

    REQUIRE(replayed.size() == original.size());
    for (std::size_t i = 0; i < replayed.size(); ++i) {
        REQUIRE(replayed[i].x == original[i].x);
        REQUIRE(replayed[i].y == original[i].y);
        REQUIRE(replayed[i].polarity == original[i].polarity);
        REQUIRE(replayed[i].timestamp == original[i].timestamp);
    }
}
