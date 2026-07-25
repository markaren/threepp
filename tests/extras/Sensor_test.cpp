// The sensor base contract: PRNG, noise model, ring buffer, rate gate.
//
// Everything here is PhysX-free on purpose. This is the machinery every future
// sensor (JointEncoder, ContactSensor, ForceTorque, GPS, ...) inherits, so a bug
// in it multiplies across the suite rather than staying in one sensor. The
// IMU's physics-truth tests live in Imu_test.cpp, which needs PhysX and so only
// builds where the SDK is available; this file runs everywhere, including CI.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/core/Object3D.hpp"
#include "threepp/extras/sensors/Sensor.hpp"

#include <cmath>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    // Minimal concrete Sensor: records what the base class hands it.
    class RecordingSensor: public Sensor {

    public:
        using Sensor::Sensor;

        std::vector<double> times;
        std::vector<double> dts;

        void sample(double dt, double simTime) override {
            dts.push_back(dt);
            times.push_back(simTime);
        }

        // resetTiming() is protected — expose it so the re-arm path is testable.
        void rearm() { resetTiming(); }
    };

    // Mean and (population) standard deviation of a sample set.
    struct Stats {
        double mean = 0.0;
        double stddev = 0.0;
    };

    Stats statsOf(const std::vector<double>& v) {
        Stats s;
        for (double x: v) s.mean += x;
        s.mean /= static_cast<double>(v.size());
        for (double x: v) s.stddev += (x - s.mean) * (x - s.mean);
        s.stddev = std::sqrt(s.stddev / static_cast<double>(v.size()));
        return s;
    }

}// namespace


// ---------------------------------------------------------------------------
// SplitMix64
// ---------------------------------------------------------------------------

TEST_CASE("SplitMix64 reproduces the reference stream") {

    // Pinned against the reference SplitMix64 (seed 0). The whole point of
    // hand-rolling this instead of using <random> is that a fixed seed gives the
    // same bits on every platform and stdlib — so the stream itself is API, and
    // a refactor that silently changes it must fail here.
    SplitMix64 rng(0);
    CHECK(rng.next() == 0xE220A8397B1DCDAFULL);
    CHECK(rng.next() == 0x6E789E6AA1B965F4ULL);
    CHECK(rng.next() == 0x06C45D188009454FULL);
    CHECK(rng.next() == 0xF88BB8A8724C81ECULL);
}

TEST_CASE("SplitMix64 is deterministic per seed") {

    SplitMix64 a(12345), b(12345), c(12346);

    bool diverged = false;
    for (int i = 0; i < 100; ++i) {
        const auto va = a.next();
        REQUIRE(va == b.next());
        if (va != c.next()) diverged = true;
    }
    CHECK(diverged);// a different seed must give a different stream
}

TEST_CASE("SplitMix64 doubles stay in [0, 1)") {

    SplitMix64 rng(99);
    double lo = 1.0, hi = 0.0, sum = 0.0;
    constexpr int n = 100000;
    for (int i = 0; i < n; ++i) {
        const double d = rng.nextDouble();
        REQUIRE(d >= 0.0);
        REQUIRE(d < 1.0);
        lo = std::min(lo, d);
        hi = std::max(hi, d);
        sum += d;
    }
    // Roughly uniform: mean near 0.5 and both tails actually reached.
    CHECK_THAT(sum / n, WithinAbs(0.5, 0.01));
    CHECK(lo < 0.01);
    CHECK(hi > 0.99);
}


// ---------------------------------------------------------------------------
// GaussianNoise
// ---------------------------------------------------------------------------

TEST_CASE("A zero noise model is a bit-exact passthrough") {

    // The physics-truth tests depend on this: a perfect sensor must return the
    // clean value unchanged, not "clean plus something that rounds to zero".
    GaussianNoise noise{NoiseModel{}};

    const Vector3 clean(1.25f, -3.5f, 9.81f);
    for (int i = 0; i < 10; ++i) {
        const Vector3 out = noise.apply(clean, 1.0 / 240.0);
        REQUIRE(out.x == clean.x);
        REQUIRE(out.y == clean.y);
        REQUIRE(out.z == clean.z);
    }
}

TEST_CASE("White noise stddev follows the continuous-time density") {

    // density / sqrt(dt) is the per-sample stddev, so halving dt raises the
    // per-sample noise by sqrt(2). This is what makes a spec-sheet figure
    // (units/sqrt(Hz)) independent of the rate the sim happens to run at.
    NoiseModel model;
    model.whiteNoiseDensity.set(0.06f, 0.06f, 0.06f);
    model.seed = 4242;

    constexpr int n = 100000;

    for (const double dt: {1.0 / 100.0, 1.0 / 400.0}) {
        GaussianNoise noise{model};
        std::vector<double> xs;
        xs.reserve(n);
        for (int i = 0; i < n; ++i) {
            xs.push_back(noise.apply(Vector3(0.f, 0.f, 0.f), dt).x);
        }

        const auto s = statsOf(xs);
        const double expected = 0.06 / std::sqrt(dt);
        INFO("dt=" << dt << " expected sigma=" << expected << " got=" << s.stddev);
        CHECK_THAT(s.stddev, WithinRel(expected, 0.02));
        CHECK_THAT(s.mean, WithinAbs(0.0, expected * 0.02));
    }
}

TEST_CASE("Bias random walk grows with the square root of elapsed time") {

    // b_k is a Wiener process driven at `randomWalk`: after t seconds the
    // ensemble stddev is rw*sqrt(t), independent of the step size used to get
    // there. Checked across independent seeds rather than along one path.
    constexpr double rw = 0.01;
    constexpr double dt = 1.0 / 200.0;
    constexpr int steps = 2000;// t = 10 s
    constexpr int runs = 400;

    std::vector<double> half, full;
    half.reserve(runs);
    full.reserve(runs);

    for (int r = 0; r < runs; ++r) {
        NoiseModel model;
        model.randomWalk.set(static_cast<float>(rw), 0.f, 0.f);
        model.seed = static_cast<std::uint64_t>(r) * 7919 + 1;
        GaussianNoise noise{model};

        for (int i = 0; i < steps; ++i) {
            noise.apply(Vector3(0.f, 0.f, 0.f), dt);
            if (i == steps / 2 - 1) half.push_back(noise.bias().x);
        }
        full.push_back(noise.bias().x);
    }

    const double sHalf = statsOf(half).stddev;
    const double sFull = statsOf(full).stddev;

    INFO("sigma(5s)=" << sHalf << " sigma(10s)=" << sFull);
    CHECK_THAT(sHalf, WithinRel(rw * std::sqrt(5.0), 0.12));
    CHECK_THAT(sFull, WithinRel(rw * std::sqrt(10.0), 0.12));
    // Doubling the time multiplies the spread by sqrt(2), not by 2.
    CHECK_THAT(sFull / sHalf, WithinRel(std::sqrt(2.0), 0.1));
}

TEST_CASE("Constant bias is a fixed offset, unaffected by dt") {

    NoiseModel model;
    model.constantBias.set(0.5f, -0.25f, 2.f);
    GaussianNoise noise{model};

    const Vector3 a = noise.apply(Vector3(1.f, 1.f, 1.f), 1.0 / 60.0);
    const Vector3 b = noise.apply(Vector3(1.f, 1.f, 1.f), 1.0 / 900.0);

    CHECK_THAT(a.x, WithinAbs(1.5f, 1e-6));
    CHECK_THAT(a.y, WithinAbs(0.75f, 1e-6));
    CHECK_THAT(a.z, WithinAbs(3.f, 1e-6));
    CHECK_THAT(b.x, WithinAbs(1.5f, 1e-6));
    CHECK_THAT(b.z, WithinAbs(3.f, 1e-6));
}

TEST_CASE("Noise is reproducible for a given seed and re-armed by reset") {

    NoiseModel model;
    model.whiteNoiseDensity.set(0.1f, 0.1f, 0.1f);
    model.randomWalk.set(0.01f, 0.01f, 0.01f);
    model.seed = 0xDEADBEEF;

    GaussianNoise a{model}, b{model};

    std::vector<Vector3> first;
    for (int i = 0; i < 50; ++i) {
        const Vector3 va = a.apply(Vector3(0.f, 0.f, 0.f), 0.01);
        const Vector3 vb = b.apply(Vector3(0.f, 0.f, 0.f), 0.01);
        REQUIRE(va.x == vb.x);
        REQUIRE(va.y == vb.y);
        REQUIRE(va.z == vb.z);
        first.push_back(va);
    }

    // reset() must rewind the stream AND clear the walked bias, otherwise an
    // episode reset in an RL loop inherits the previous episode's drift.
    a.reset(model);
    CHECK(a.bias().x == 0.f);
    for (int i = 0; i < 50; ++i) {
        const Vector3 v = a.apply(Vector3(0.f, 0.f, 0.f), 0.01);
        REQUIRE(v.x == first[i].x);
        REQUIRE(v.z == first[i].z);
    }
}

TEST_CASE("A non-positive dt adds no white noise and never produces NaN") {

    // Guards the 1/sqrt(dt) term. A zero-length interval can reach a sensor via
    // a paused world or a rate gate that fires twice on the same sim time.
    NoiseModel model;
    model.whiteNoiseDensity.set(1.f, 1.f, 1.f);
    model.randomWalk.set(1.f, 1.f, 1.f);
    model.constantBias.set(0.25f, 0.f, 0.f);
    GaussianNoise noise{model};

    for (const double dt: {0.0, -1.0}) {
        const Vector3 v = noise.apply(Vector3(3.f, 4.f, 5.f), dt);
        INFO("dt=" << dt);
        REQUIRE(std::isfinite(v.x));
        REQUIRE(std::isfinite(v.y));
        REQUIRE(std::isfinite(v.z));
        CHECK_THAT(v.x, WithinAbs(3.25f, 1e-6));// constant bias only
        CHECK_THAT(v.y, WithinAbs(4.f, 1e-6));
    }
}


// ---------------------------------------------------------------------------
// SensorRing
// ---------------------------------------------------------------------------

TEST_CASE("SensorRing drains oldest-first and empties") {

    SensorRing<int> ring(8);
    CHECK(ring.empty());
    CHECK(ring.capacity() == 8);

    for (int i = 0; i < 5; ++i) ring.push(i);
    CHECK(ring.size() == 5);

    std::vector<int> out;
    ring.drain(out);
    REQUIRE(out.size() == 5);
    for (int i = 0; i < 5; ++i) CHECK(out[i] == i);

    CHECK(ring.empty());
    ring.drain(out);
    CHECK(out.empty());
}

TEST_CASE("SensorRing drops the oldest sample on overflow") {

    // A stalled reader must never stall the physics step, so the newest sample
    // always lands and the oldest is the one that goes.
    SensorRing<int> ring(4);
    for (int i = 0; i < 10; ++i) ring.push(i);

    CHECK(ring.size() == 4);

    std::vector<int> out;
    ring.drain(out);
    REQUIRE(out.size() == 4);
    CHECK(out[0] == 6);
    CHECK(out[3] == 9);
}

TEST_CASE("SensorRing latest() survives drain but not clear") {

    SensorRing<int> ring(4);
    CHECK_FALSE(ring.latest().has_value());

    ring.push(1);
    ring.push(2);
    CHECK(ring.latest().value() == 2);

    std::vector<int> out;
    ring.drain(out);
    CHECK(ring.empty());
    CHECK(ring.latest().value() == 2);// a snapshot, not a queue entry

    ring.clear();
    CHECK_FALSE(ring.latest().has_value());
}

TEST_CASE("SensorRing survives a zero capacity request") {

    SensorRing<int> ring(0);
    CHECK(ring.capacity() == 1);// clamped, so the modulo never divides by zero

    ring.push(7);
    ring.push(8);
    CHECK(ring.size() == 1);
    CHECK(ring.latest().value() == 8);
}


// ---------------------------------------------------------------------------
// Sensor rate gate
// ---------------------------------------------------------------------------

TEST_CASE("An ungated sensor samples every substep") {

    auto node = Object3D::create();
    RecordingSensor s(*node, 0.0);

    constexpr double dt = 1.0 / 240.0;
    for (int i = 0; i < 100; ++i) s.tick(dt, (i + 1) * dt);

    REQUIRE(s.times.size() == 100);
    for (double d: s.dts) CHECK_THAT(d, WithinAbs(dt, 1e-9));
}

TEST_CASE("A rate-gated sensor holds its requested average rate") {

    // The gate has to schedule against a fixed period, not against the time the
    // previous sample happened to land on. Chasing the actual emission time
    // rounds the interval UP to a whole substep every time: at 240 Hz physics a
    // 100 Hz request (period 10 ms) would emit every 3rd substep = 12.5 ms =
    // 80 Hz, a 20% rate error, and an IMU that quietly delivers 80 Hz breaks
    // any lock-step consumer that was promised 100.
    auto node = Object3D::create();
    RecordingSensor s(*node, 100.0);

    constexpr double dt = 1.0 / 240.0;
    constexpr int substeps = 480;// 2 seconds
    for (int i = 0; i < substeps; ++i) s.tick(dt, (i + 1) * dt);

    INFO("emitted " << s.times.size() << " samples in 2 s");
    CHECK(s.times.size() >= 198);
    CHECK(s.times.size() <= 202);

    // Jitter is bounded by one substep (a sample can only land on a substep
    // boundary), but it must not accumulate into drift.
    const double span = s.times.back() - s.times.front();
    const double meanPeriod = span / static_cast<double>(s.times.size() - 1);
    CHECK_THAT(meanPeriod, WithinAbs(0.01, 1e-4));
}

TEST_CASE("A rate-gated sensor emits immediately, then gates") {

    auto node = Object3D::create();
    RecordingSensor s(*node, 50.0);// period 20 ms

    constexpr double dt = 0.001;
    s.tick(dt, dt);
    REQUIRE(s.times.size() == 1);// first tick always samples

    // The period anchors to the first sample (t = 1 ms), so the next is due at
    // 21 ms — nothing between fires.
    for (int i = 1; i < 20; ++i) s.tick(dt, (i + 1) * dt);
    CHECK(s.times.size() == 1);

    s.tick(dt, 0.021);
    REQUIRE(s.times.size() == 2);
    CHECK_THAT(s.dts[1], WithinAbs(0.020, 1e-9));// full period, not the substep dt
}

TEST_CASE("Rate gating reports the true elapsed interval to sample()") {

    // A sensor running slower than the physics rate still has to finite-
    // difference over the real interval, not over the substep dt.
    auto node = Object3D::create();
    RecordingSensor s(*node, 60.0);

    constexpr double dt = 1.0 / 600.0;
    for (int i = 0; i < 600; ++i) s.tick(dt, (i + 1) * dt);

    REQUIRE(s.dts.size() > 2);
    // Skip the first (it is handed the substep dt by definition) and check the
    // rest sum to the covered span with no gaps.
    double sum = 0.0;
    for (std::size_t i = 1; i < s.dts.size(); ++i) sum += s.dts[i];
    CHECK_THAT(sum, WithinAbs(s.times.back() - s.times.front(), 1e-9));
}

TEST_CASE("A sensor gated faster than the physics rate samples every substep") {

    // 1 kHz requested, 240 Hz physics: the gate must not try to burst-catch-up
    // by firing repeatedly on the same substep, and must not fall permanently
    // behind either.
    auto node = Object3D::create();
    RecordingSensor s(*node, 1000.0);

    constexpr double dt = 1.0 / 240.0;
    for (int i = 0; i < 240; ++i) s.tick(dt, (i + 1) * dt);

    CHECK(s.times.size() == 240);
}

TEST_CASE("resetTiming re-arms the gate and the first-sample path") {

    auto node = Object3D::create();
    RecordingSensor s(*node, 50.0);

    constexpr double dt = 0.001;
    for (int i = 0; i < 100; ++i) s.tick(dt, (i + 1) * dt);
    const auto before = s.times.size();
    REQUIRE(before >= 5);

    s.rearm();
    s.times.clear();
    s.dts.clear();

    // After a re-arm the very next tick samples again, whatever the clock says,
    // and it is handed the substep dt rather than a huge interval spanning the
    // reset.
    s.tick(dt, 0.101);
    REQUIRE(s.times.size() == 1);
    CHECK_THAT(s.dts[0], WithinAbs(dt, 1e-9));
}
