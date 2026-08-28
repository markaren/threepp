// Joint encoder truth.
//
// The interesting assertions are about QUANTIZATION, because that is what an
// encoder adds over Articulation's exact jointPositions(): readings land on tick
// boundaries, differentiated velocity therefore lands on multiples of one tick
// per sample interval, and motion smaller than a tick reads as no motion at all.
// Those three are what make a controller tuned in sim survive on hardware.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

    constexpr float kSubstep = 1.f / 240.f;

    PhysxWorld::Settings fixedStep() {
        PhysxWorld::Settings s;
        s.fixedTimestep = kSubstep;
        s.maxSubSteps = 1;
        return s;
    }

    std::shared_ptr<Mesh> box(float x, float y, float z, float size = 0.2f) {
        auto m = Mesh::create(BoxGeometry::create(size, size, size), MeshBasicMaterial::create());
        m->position.set(x, y, z);
        return m;
    }

    // A one-joint pendulum: fixed root at the origin, one child link hanging off a
    // free revolute joint about Z anchored at the origin. Gravity swings it, so
    // the joint angle sweeps a wide continuous range — good material for a
    // quantization test.
    struct Pendulum {
        std::unique_ptr<Articulation> art;
        std::shared_ptr<Mesh> rootMesh;
        std::shared_ptr<Mesh> armMesh;
        ArticulationLink root{nullptr, nullptr, nullptr, ::physx::PxTransform(::physx::PxIdentity)};
        ArticulationLink arm{nullptr, nullptr, nullptr, ::physx::PxTransform(::physx::PxIdentity)};
    };

    // `horizontal` starts the arm along +X (maximum gravity torque, so it swings);
    // otherwise it hangs straight down along -Y and barely moves.
    Pendulum makePendulum(PhysxWorld& world, bool horizontal) {
        Pendulum p;
        p.art = std::make_unique<Articulation>(world, /*fixedBase*/ true,
                                               /*solverPositionIters*/ 8,
                                               /*disableSelfCollision*/ true);
        p.rootMesh = box(0.f, 0.f, 0.f);
        p.armMesh = horizontal ? box(1.f, 0.f, 0.f) : box(0.f, -1.f, 0.f);

        p.root = p.art->addLink(nullptr, *p.rootMesh, 1000.f, {0, 0, 1}, {0, 0, 0},
                                false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);
        p.arm = p.art->addLink(&p.root, *p.armMesh, 1000.f, {0, 0, 1}, {0, 0, 0},
                               false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);
        p.art->finalize();
        return p;
    }

    void stepFor(PhysxWorld& world, int substeps) {
        for (int i = 0; i < substeps; ++i) world.step(kSubstep);
    }

}// namespace


TEST_CASE("An ideal encoder reports the exact joint angle") {

    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    JointEncoder enc(*p.armMesh, p.arm);// resolution 0, no noise
    world.registerSensor(&enc);

    stepFor(world, 60);

    const auto s = enc.latest();
    REQUIRE(s.has_value());
    // Sampled inside the last substep, and nothing has stepped since, so it must
    // match the joint exactly — bit-for-bit, since a zero NoiseModel is a
    // passthrough and resolution 0 skips quantization entirely.
    CHECK(s->position == p.arm.jointPosition());

    // And it actually moved, or the above is vacuous.
    CHECK(std::abs(s->position) > 0.1f);

    world.unregisterSensor(&enc);
}

TEST_CASE("A quantized encoder reports whole ticks only") {

    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    constexpr float res = 0.05f;// rad per tick
    JointEncoder enc(*p.armMesh, p.arm);
    enc.resolution = res;
    world.registerSensor(&enc);

    stepFor(world, 240);

    std::vector<JointSample> samples;
    enc.drain(samples);
    REQUIRE(samples.size() == 240);

    bool moved = false;
    for (const auto& s: samples) {
        const float ticks = s.position / res;
        INFO("position " << s.position << " = " << ticks << " ticks");
        REQUIRE_THAT(ticks - std::round(ticks), WithinAbs(0.f, 1e-3f));
        if (std::abs(s.position) > 0.5f) moved = true;
    }
    CHECK(moved);

    // The quantized reading never strays more than half a tick from truth.
    CHECK_THAT(samples.back().position, WithinAbs(p.arm.jointPosition(), res * 0.5f + 1e-4f));

    world.unregisterSensor(&enc);
}

TEST_CASE("Differentiated velocity is quantized to one tick per interval") {

    // The characteristic encoder artifact: velocity can only be an integer
    // number of ticks per sample interval. A controller differentiating encoder
    // counts sees this staircase, never a smooth derivative.
    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    constexpr float res = 0.05f;
    JointEncoder enc(*p.armMesh, p.arm);
    enc.resolution = res;
    REQUIRE(enc.differentiateVelocity);// the default
    world.registerSensor(&enc);

    stepFor(world, 240);

    std::vector<JointSample> samples;
    enc.drain(samples);
    REQUIRE(samples.size() == 240);

    const float step = res / kSubstep;// one tick per substep = 12 rad/s
    bool nonZero = false;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const float n = samples[i].velocity / step;
        INFO("sample " << i << " v=" << samples[i].velocity << " = " << n << " ticks/interval");
        REQUIRE_THAT(n - std::round(n), WithinAbs(0.f, 1e-3f));
        if (std::abs(samples[i].velocity) > 0.f) nonZero = true;
    }
    CHECK(nonZero);

    // No startup spike: the first sample has no predecessor to difference.
    CHECK(samples.front().velocity == 0.f);

    world.unregisterSensor(&enc);
}

TEST_CASE("Sub-tick motion reads as no motion") {

    // A coarse encoder on a slowly-moving joint reports exact zeros — not the
    // small real motion the simulator has. This is the behaviour that makes a
    // real robot's velocity estimate dead at low speed.
    //
    // Driven from horizontal (so the joint is provably moving under gravity) but
    // stopped well before it covers half a tick, which is the only way to tell
    // "the encoder suppressed the motion" from "there was no motion".
    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    JointEncoder enc(*p.armMesh, p.arm);
    enc.resolution = 1.0f;// absurdly coarse: ~57 degrees per tick
    world.registerSensor(&enc);

    stepFor(world, 20);// ~83 ms of swing

    std::vector<JointSample> samples;
    enc.drain(samples);
    REQUIRE(samples.size() == 20);

    for (const auto& s: samples) {
        INFO("t=" << s.t << " pos=" << s.position << " vel=" << s.velocity);
        REQUIRE(s.position == 0.f);
        REQUIRE(s.velocity == 0.f);
    }

    // The joint really did move — strictly more than nothing, strictly less than
    // half a tick — so the zeros above are suppression, not absence of motion.
    const float trueAngle = std::abs(p.arm.jointPosition());
    INFO("true |angle| = " << trueAngle << " rad, half-tick = " << enc.resolution * 0.5f);
    CHECK(trueAngle > 1e-3f);
    CHECK(trueAngle < enc.resolution * 0.5f);
    CHECK(std::abs(p.arm.jointVelocity()) > 1e-2f);// and moving, not just displaced

    world.unregisterSensor(&enc);
}

TEST_CASE("An encoder can report the simulator's true velocity instead") {

    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    JointEncoder enc(*p.armMesh, p.arm);
    enc.differentiateVelocity = false;// resolver / tachometer stand-in
    world.registerSensor(&enc);

    stepFor(world, 60);

    const auto s = enc.latest();
    REQUIRE(s.has_value());
    CHECK(s->velocity == p.arm.jointVelocity());// zero noise => passthrough
    CHECK(std::abs(s->velocity) > 0.1f);

    world.unregisterSensor(&enc);
}

TEST_CASE("countsPerRev sets the resolution") {

    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    JointEncoder enc(*p.armMesh, p.arm);
    enc.setCountsPerRev(4096);
    CHECK_THAT(enc.resolution, WithinRel(math::TWO_PI / 4096.f, 1e-5f));

    CHECK_THROWS_AS(enc.setCountsPerRev(0), std::invalid_argument);
    CHECK_THROWS_AS(enc.setCountsPerRev(-8), std::invalid_argument);
}

TEST_CASE("Encoder noise is reproducible and respects the seed") {

    const auto run = [] {
        PhysxWorld world(fixedStep());
        auto p = makePendulum(world, /*horizontal*/ true);

        JointEncoder enc(*p.armMesh, p.arm);
        enc.positionNoise.whiteNoiseDensity.set(0.002f, 0.f, 0.f);
        enc.positionNoise.seed = 0xC0FFEE;
        enc.reset();
        world.registerSensor(&enc);

        stepFor(world, 60);
        std::vector<JointSample> out;
        enc.drain(out);
        world.unregisterSensor(&enc);
        return out;
    };

    const auto a = run();
    const auto b = run();
    REQUIRE(a.size() == 60);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        INFO("sample " << i);
        REQUIRE(a[i].position == b[i].position);
        REQUIRE(a[i].velocity == b[i].velocity);
    }

    // Noise actually perturbed the reading away from the exact joint angle.
    PhysxWorld clean(fixedStep());
    auto p = makePendulum(clean, /*horizontal*/ true);
    JointEncoder ideal(*p.armMesh, p.arm);
    clean.registerSensor(&ideal);
    stepFor(clean, 60);
    std::vector<JointSample> exact;
    ideal.drain(exact);
    clean.unregisterSensor(&ideal);

    bool differs = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].position != exact[i].position) differs = true;
    }
    CHECK(differs);
}

TEST_CASE("A rate-gated encoder runs on its own cadence") {

    PhysxWorld world(fixedStep());// 240 Hz
    auto p = makePendulum(world, /*horizontal*/ true);

    JointEncoder enc(*p.armMesh, p.arm, /*rateHz*/ 50.0);
    world.registerSensor(&enc);

    stepFor(world, 240);// 1 second

    std::vector<JointSample> samples;
    enc.drain(samples);
    INFO("emitted " << samples.size() << " samples at 50 Hz over 1 s");
    CHECK(samples.size() >= 49);
    CHECK(samples.size() <= 51);

    // Differentiation must use the ENCODER's interval, not the physics substep:
    // over ~20 ms the pendulum moves ~5x further than over one substep, and a
    // velocity computed against the wrong dt would be off by that factor.
    const auto& s = samples.back();
    CHECK_THAT(s.velocity, WithinRel(p.arm.jointVelocity(), 0.25f));

    world.unregisterSensor(&enc);
}

TEST_CASE("Attaching an encoder to the root link throws") {

    PhysxWorld world(fixedStep());
    auto p = makePendulum(world, /*horizontal*/ true);

    CHECK_THROWS_AS(JointEncoder(*p.rootMesh, p.root), std::invalid_argument);
}
