
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/physx/PhysxParticles.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    // PBD particles are CUDA-only with no CPU path, so on a machine without a
    // GPU (ci-linux-physx) there is nothing to exercise. Same shape as
    // SoftBody_test::gpuWorld: return null, print why, and let the test pass —
    // the alternative is a red CI that says nothing about the code.
    std::unique_ptr<PhysxWorld> gpuWorld() {

        PhysxWorld::Settings settings;
        settings.enableGpuDynamics = true;
        settings.gpuHeapCapacityMB = 512;
        try {
            return std::make_unique<PhysxWorld>(settings);
        } catch (const std::exception&) {
            return nullptr;// no CUDA device here
        }
    }

    constexpr float kSpacing = 0.05f;
    constexpr float kBeltY = 1.0f;
    constexpr float kBeltHalfLen = 3.0f;
    constexpr float kBeltWidth = 1.0f;
    constexpr float kBeltSpeed = 1.5f;
    constexpr int kSteps = 240;// 4 s at the world's 1/60 fixed step

    conveyor::ConveyorSpec beltSpec() {

        conveyor::ConveyorSpec s;
        s.waypoints = {conveyor::Waypoint{Vector3(-kBeltHalfLen, kBeltY, 0.f)},
                       conveyor::Waypoint{Vector3(+kBeltHalfLen, kBeltY, 0.f)}};
        s.width = kBeltWidth;
        s.speed = kBeltSpeed;
        s.smooth = false;
        s.frame = false;// visual only; the test draws nothing
        for (int side = -1; side <= 1; side += 2) {
            conveyor::WallSpec w;
            w.height = 0.35f;
            w.points = {Vector3(-kBeltHalfLen, kBeltY, float(side) * kBeltWidth * 0.5f),
                        Vector3(+kBeltHalfLen, kBeltY, float(side) * kBeltWidth * 0.5f)};
            s.walls.push_back(std::move(w));
        }
        return s;
    }

    // A bed of grains resting on the inlet half of the belt: a lattice a hair
    // apart, so nothing starts interpenetrating (PBD depenetrates an overlap
    // hard, and this test would then measure the eruption).
    std::vector<Vector3> bed(unsigned count) {

        // 3 m x 0.9 m footprint, so 5000 grains are ~6 layers (0.3 m) deep and
        // stay inside the 0.35 m guides — a bed that overtops them is a jam, and
        // a jam is not what this test is measuring.
        const float d = kSpacing * 1.06f;
        const auto nx = unsigned(3.0f / d);
        const auto nz = unsigned(kBeltWidth * 0.9f / d);
        std::vector<Vector3> out;
        out.reserve(count);
        for (unsigned y = 0; out.size() < count; ++y)
            for (unsigned x = 0; x < nx && out.size() < count; ++x)
                for (unsigned z = 0; z < nz && out.size() < count; ++z)
                    out.emplace_back(-1.4f + (float(x) - float(nx - 1) * 0.5f) * d,
                                     kBeltY + 0.03f + float(y) * d,
                                     (float(z) - float(nz - 1) * 0.5f) * d);
        return out;
    }

    struct Run {
        float carried = 0.f;// mean displacement along travel (+X), metres
        float minY = 0.f;
        unsigned bad = 0;// non-finite components
        unsigned n = 0;
    };

    float meanX(const ::physx::PxVec4* p, unsigned n) {

        double sx = 0;
        unsigned good = 0;
        for (unsigned i = 0; i < n; ++i) {
            if (!std::isfinite(p[i].x)) continue;
            sx += p[i].x;
            ++good;
        }
        return good ? float(sx / good) : 0.f;
    }

    // One complete pour-and-convey run. `speedScale` is ConveyorPhysics' own
    // live multiplier, so 0 gives the CONTROL: the identical scene with the belt
    // stopped. Everything is scoped so it tears down in the required order —
    // the belt and the particle system both borrow the world and both release
    // actors / device memory through it.
    Run runBelt(unsigned count, float speedScale) {

        using namespace ::physx;

        Run r;
        auto world = gpuWorld();
        REQUIRE(world);

        PbdParticles::Settings ps;
        ps.spacing = kSpacing;
        ps.solverIterations = 8;
        PbdParticles particles(*world, ps);

        PbdParticles::MaterialSpec spec;
        spec.friction = 1.2f;
        spec.damping = 0.2f;
        auto& grains = particles.addGroup(count, spec);

        auto* ground = world->physics().createMaterial(0.9f, 0.9f, 0.f);
        world->addStatic(PxBoxGeometry(12.f, 0.2f, 12.f),
                         PxTransform(PxVec3(0.f, -0.2f, 0.f)), ground);

        conveyor::ConveyorPhysics belt(*world, {beltSpec()});
        belt.speedScale = speedScale;

        const auto seeded = bed(count);
        r.n = grains.emit(seeded.data(), unsigned(seeded.size()), Vector3(0.f, 0.f, 0.f), 0.02f);
        REQUIRE(r.n == count);

        particles.pull();
        const float x0 = meanX(grains.positions(), r.n);

        for (int i = 0; i < kSteps; ++i) world->step(1.f / 60.f);
        particles.pull();

        r.carried = meanX(grains.positions(), r.n) - x0;
        const auto* p = grains.positions();
        r.minY = p[0].y;
        for (unsigned i = 0; i < r.n; ++i) {
            if (!std::isfinite(p[i].x) || !std::isfinite(p[i].y) || !std::isfinite(p[i].z)) {
                ++r.bad;
                continue;
            }
            r.minY = std::min(r.minY, p[i].y);
        }
        return r;
    }

}// namespace


// The pin: a GPU PBD bed on a running belt is CARRIED, and the belt is what
// carries it. An absolute threshold alone would be a tuning artefact — the top
// of a deep bed always lags the surface — so the same scene is run twice, once
// driven and once with ConveyorPhysics::speedScale at zero. The difference
// between those two runs is the coupling this demo exists to demonstrate.
//
// Also pinned here, because both were real failures during development:
//   • no non-finite positions (an overlapping emit used to erupt);
//   • nothing sinks through the floor slab (fixed by raising
//     PxParticleFlag::eENABLE_SPECULATIVE_CCD in PbdParticles).
TEST_CASE("a PBD granular bed is carried by a conveyor belt", "[physx][gpu]") {

    if (!gpuWorld()) {
        std::cout << "[skip] no CUDA device - PxPBDParticleSystem has no CPU path, "
                     "so PBD particles are not exercised"
                  << std::endl;
        return;
    }

    constexpr unsigned kCount = 5000;
    const Run driven = runBelt(kCount, 1.f);
    const Run idle = runBelt(kCount, 0.f);

    std::cout << "  driven: carried " << driven.carried << " m, minY " << driven.minY
              << "\n  idle:   carried " << idle.carried << " m, minY " << idle.minY << std::endl;

    CHECK(driven.bad == 0);
    CHECK(idle.bad == 0);

    // The belt's top face is at kBeltY and the floor's at 0; a resting grain
    // sits one radius above whichever it is on. Anything materially below the
    // floor has tunnelled out of the world.
    const float radius = 0.5f * kSpacing;
    CHECK(driven.minY > -radius);
    CHECK(idle.minY > -radius);

    // 4 s of a 1.5 m/s belt is 6 m of surface travel. A bed 0.3 m deep slips,
    // so this is deliberately a fraction of that rather than a fit to the
    // measurement — it has to fail on a belt that has stopped conveying, not
    // on one that conveys 20% slower after a solver change.
    CHECK(driven.carried > 0.25f * kBeltSpeed * (float(kSteps) / 60.f));
    // The control barely moves: a stopped belt is a level surface, so the bed
    // only settles and spreads a little.
    CHECK(idle.carried < 0.30f);
    CHECK(driven.carried > 4.f * std::abs(idle.carried));
}
