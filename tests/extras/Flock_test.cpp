// Flock_test — the contracts Flock.hpp states in prose, defended by CI.
//
// The headline claim is DETERMINISM: "bit-identical for the same binary, the
// same seed, and the same dt sequence" (Flock.hpp banner). Until now that
// lived only as a comment and a demo-local --selftest nothing runs
// automatically. These cases are deliberately EXACT-equality on floats — the
// contract is bit-identity within one binary, so an epsilon would test a
// weaker promise than the one the header makes.
//
// flock_demo --selftest remains the richer behavioural soak (3600-step
// passes, startle containment against baked scenery); this file keeps the
// fast, scenery-light versions of the properties that must never regress.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/fauna/Flock.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>

using namespace threepp;

namespace {

    constexpr float kDt = 1.f / 60.f;

    Flock::Params testParams() {
        Flock::Params p;
        p.seed = 42u;
        p.birdCount = 12;
        p.home.set(0.f, 10.f, 0.f);
        p.roamRadius = 30.f;
        // Perch quickly so a short soak sees the whole state machine:
        // Cruise → Approach → Flare → Perched → Launch.
        p.perchIntervalMin = 1.f;
        p.perchIntervalMax = 4.f;
        p.restIntervalMin = 1.f;
        p.restIntervalMax = 3.f;
        return p;
    }

    void addRailPerches(Flock& f) {
        // Authored perches — no scene, no bake, no BVH. setPerches/addPerch is
        // the designed escape hatch, which makes it the cheapest deterministic
        // fixture the state machine can land on. SPREAD OUT on purpose: with
        // 12 birds and a 0.55 committed cap, six birds approach at once, and
        // spots packed closer than ~2× separationDistance turn the final-metre
        // capture into a shoving match no approach survives.
        for (int i = 0; i < 8; ++i) {
            const float a = static_cast<float>(i) * (math::TWO_PI / 8.f);
            f.addPerch({14.f * std::cos(a), 2.f + 0.5f * static_cast<float>(i % 3),
                        14.f * std::sin(a)},
                       {0.f, 1.f, 0.f}, true);
        }
    }

    // One shared script for the replay pair: identical dt sequence (three
    // periods, so dtSmoothed never settles), identical mid-run startle.
    void runScript(Flock& f) {
        constexpr float dts[3] = {1.f / 60.f, 1.f / 45.f, 1.f / 90.f};
        for (int step = 0; step < 900; ++step) {
            if (step == 400) f.startle({0.f, 10.f, 0.f}, 1e9f, 1.f);
            f.update(dts[step % 3]);
        }
    }

    bool bitIdentical(const Flock& a, const Flock& b) {
        if (a.birdCount() != b.birdCount()) return false;
        for (int i = 0; i < a.birdCount(); ++i) {
            const Vector3 &pa = a.birdPosition(i), &pb = b.birdPosition(i);
            const Vector3 &va = a.birdVelocity(i), &vb = b.birdVelocity(i);
            // Exact on purpose — see the file banner.
            if (pa.x != pb.x || pa.y != pb.y || pa.z != pb.z) return false;
            if (va.x != vb.x || va.y != vb.y || va.z != vb.z) return false;
            if (a.stateOf(i) != b.stateOf(i)) return false;
        }
        return true;
    }

}// namespace

TEST_CASE("Flock: same seed + same dt sequence replays bit-identically") {

    auto a = Flock::create(testParams());
    auto b = Flock::create(testParams());
    addRailPerches(*a);
    addRailPerches(*b);

    runScript(*a);
    runScript(*b);

    REQUIRE(bitIdentical(*a, *b));
    // A different seed must actually diverge, or the check above proves
    // nothing (a flock pinned at home would also "replay" perfectly).
    auto c = Flock::create([] { auto p = testParams(); p.seed = 43u; return p; }());
    addRailPerches(*c);
    runScript(*c);
    REQUIRE_FALSE(bitIdentical(*a, *c));
}

TEST_CASE("PerchIndex: blocking and amortised bakes produce identical tables") {

    Scene scene;
    auto mat = MeshStandardMaterial::create();
    for (int i = 0; i < 3; ++i) {
        auto box = Mesh::create(BoxGeometry::create(4.f, 2.f + i, 4.f), mat);
        box->position.set(-8.f + 8.f * static_cast<float>(i), 0.5f * (2.f + i), 0.f);
        scene.add(box);
    }

    fauna::PerchIndex blocking;
    fauna::PerchIndex::Params pp;
    blocking.bakeBlocking(scene, pp, nullptr, nullptr);

    fauna::PerchIndex amortised;
    amortised.begin(scene, pp, nullptr, nullptr);
    int steps = 0;
    while (!amortised.step()) {
        REQUIRE(++steps < 100000);// a bake that never completes is its own failure
    }

    REQUIRE(blocking.spots().size() > 0);
    // PerchSpot's defaulted operator== is exact float compare — the banner's
    // "bit-identical perch tables" claim, taken literally.
    REQUIRE(blocking.spots() == amortised.spots());
}

TEST_CASE("Flock: birds land, stay contained, and never NaN in a short soak") {

    auto flock = Flock::create(testParams());
    addRailPerches(*flock);
    const auto& p = flock->params();

    bool everPerched = false;
    bool allFinite = true;
    float worstRadius = 0.f;
    for (int step = 0; step < 2400; ++step) {
        flock->update(kDt);
        for (int i = 0; i < flock->birdCount(); ++i) {
            const Vector3& pos = flock->birdPosition(i);
            allFinite = allFinite && std::isfinite(pos.x) && std::isfinite(pos.y) &&
                        std::isfinite(pos.z);
            everPerched = everPerched || flock->stateOf(i) == Flock::BirdState::Perched;
            const float dx = pos.x - p.home.x, dz = pos.z - p.home.z;
            worstRadius = std::max(worstRadius, std::sqrt(dx * dx + dz * dz));
        }
    }

    REQUIRE(allFinite);
    REQUIRE(everPerched);// the landing pipeline fired — the bug --selftest exists to catch
    REQUIRE(worstRadius <= 1.5f * p.roamRadius);
}
