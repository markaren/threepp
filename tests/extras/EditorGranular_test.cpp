// The granular half of the particle-authoring pipeline, end to end and with no
// editor application anywhere: a chute and a floor authored as scene content,
// played through the real PhysicsPlaySession + GranularPlaySession in their
// registration order, and asked the two questions that make it a feature —
// grains ARRIVE, and they land ON the floor rather than through it. The same
// document is then played a second time, because "an episode starts from zero"
// is the property that lets a scene be played twice.
//
// The grain positions are read back off the InstancedMesh the session built, so
// the assertion covers the whole chain — escalated GPU world, PBD solver, chute
// emitter, pull(), visual — rather than just the solver's own numbers.
//
// [physx][gpu]: PxPBDParticleSystem is a CUDA-only PhysX feature with no CPU
// path, so a machine without a device has nothing here to exercise. Same shape
// as PbdParticles_test: probe, print why, and pass.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/GranularPlaySession.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr float kDt = 1.f / 60.f;
    constexpr float kSpacing = 0.06f;
    constexpr int kCapacity = 30000;// modest on purpose: this is not a bench
    constexpr float kRate = 5000.f;
    constexpr float kChuteY = 2.f;

    bool cudaAvailable() {

        PhysxWorld::Settings settings;
        settings.enableGpuDynamics = true;
        try {
            PhysxWorld world(settings);
            return world.cudaContextManager() != nullptr;
        } catch (const std::exception&) {
            return false;// no CUDA device here
        }
    }

    // A static slab whose TOP FACE is y = 0, and a chute standing over it.
    // Returns the chute node.
    Object3D* authorScene(Scene& scene) {

        auto floor = Mesh::create(BoxGeometry::create(8.f, 0.4f, 8.f),
                                  MeshStandardMaterial::create());
        floor->name = "Floor";
        floor->position.y = -0.2f;
        PhysicsConfig slab;
        slab.enabled = true;
        slab.body = PhysicsConfig::Body::Static;
        slab.shape = PhysicsConfig::Shape::Box;
        slab.write(*floor);
        scene.add(floor);

        auto chute = ObjectFactory::createGranular(scene);
        chute->position.set(0.f, kChuteY, 0.f);
        GranularConfig config;
        config.spacing = kSpacing;
        config.capacity = kCapacity;
        config.rate = kRate;
        config.write(*chute);
        auto* node = chute.get();
        scene.add(chute);
        return node;
    }

    // The visual the session added to the scene root. Null once it has stopped.
    InstancedMesh* grainMesh(Scene& scene) {

        InstancedMesh* found = nullptr;
        scene.traverse([&](Object3D& object) {
            if (auto* mesh = object.as<InstancedMesh>()) found = mesh;
        });
        return found;
    }

    struct Pile {
        float minY = 0.f;
        float maxY = 0.f;
        unsigned settled = 0;// within half a metre of the floor
        unsigned bad = 0;    // non-finite
    };

    // Reads the live prefix of the instance matrices — the translation of every
    // grain the session claims is alive.
    Pile measure(const InstancedMesh& mesh, unsigned n) {

        Pile pile;
        const auto& e = mesh.instanceMatrix()->array();
        pile.minY = pile.maxY = e[13];
        for (unsigned i = 0; i < n; ++i) {
            const float x = e[std::size_t(i) * 16 + 12];
            const float y = e[std::size_t(i) * 16 + 13];
            const float z = e[std::size_t(i) * 16 + 14];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                ++pile.bad;
                continue;
            }
            pile.minY = std::min(pile.minY, y);
            pile.maxY = std::max(pile.maxY, y);
            if (y < 0.5f) ++pile.settled;
        }
        return pile;
    }

}// namespace


TEST_CASE("an authored granular chute pours grains that land on the floor", "[physx][gpu]") {

    if (!cudaAvailable()) {
        std::cout << "[skip] no CUDA device - PxPBDParticleSystem has no CPU path, "
                     "so authored granular chutes are not exercised"
                  << std::endl;
        return;
    }

    Scene scene;
    auto* chute = authorScene(scene);
    REQUIRE(chute != nullptr);

    PhysicsPlaySession physics;
    GranularPlaySession granular;
    granular.setPhysics(&physics);
    granular.setLogger([](const std::string& message) { std::cout << "  " << message << std::endl; });
    physics.setLogger([](const std::string& message) { std::cout << "  " << message << std::endl; });

    // Registration order, which is also what makes the emit legal: the physics
    // session steps the world, and this one emits between those steps.
    physics.start(scene);
    granular.start(scene);

    // The chute is what escalated the world: nothing here asked for GPU
    // dynamics, and PBD cannot run without it.
    REQUIRE(physics.world() != nullptr);
    CHECK(physics.gpuAvailable());
    CHECK(physics.world()->cudaContextManager() != nullptr);
    CHECK(physics.world()->settings().gpuHeapCapacityMB >= 512);
    CHECK(physics.bodyCount() == 1);// the floor; a chute is never a rigid body

    REQUIRE_FALSE(granular.declined());
    CHECK(granular.granularNodeCount() == 1);
    CHECK(granular.groupCount() == 1);
    CHECK(granular.activeGrainCount() == 0);// nothing has been poured yet

    // Headless: no renderer, so Visual::Auto resolves to the InstancedMesh.
    CHECK_FALSE(granular.fieldVisuals());
    REQUIRE(grainMesh(scene) != nullptr);

    unsigned afterOneSecond = 0;
    for (int i = 0; i < 120; ++i) {// 2 s at the world's fixed step
        physics.update(kDt);
        granular.update(kDt);
        if (i == 59) afterOneSecond = granular.activeGrainCount();
    }

    const unsigned poured = granular.activeGrainCount();
    std::cout << "  poured " << afterOneSecond << " grains in 1 s, " << poured << " in 2 s"
              << std::endl;
    CHECK(afterOneSecond > 0);
    CHECK(poured > afterOneSecond);
    // A fractional accumulator over 120 steps of 5000/s, minus nothing: the
    // group is nowhere near its 30k capacity.
    CHECK(poured < static_cast<unsigned>(kCapacity));

    auto* mesh = grainMesh(scene);
    REQUIRE(mesh != nullptr);
    CHECK(mesh->count() >= poured);// stepped in 4096 blocks, so never less

    const auto pile = measure(*mesh, poured);
    std::cout << "  pile: minY " << pile.minY << ", maxY " << pile.maxY << ", settled "
              << pile.settled << std::endl;
    CHECK(pile.bad == 0);
    // A resting grain sits one radius above the surface it landed on, and the
    // floor's top face is y = 0. Anything materially below that has tunnelled.
    const float radius = 0.5f * kSpacing;
    CHECK(pile.minY > -radius);
    // The pour reaches the floor rather than hanging in the air: everything
    // emitted before t = 1.36 s has fallen the two metres by now.
    CHECK(pile.settled > poured / 4);
    // And it is still coming out of the chute at the top.
    CHECK(pile.maxY > kChuteY - 0.5f);

    granular.stop();
    physics.stop();

    // The visuals are the session's, not the document's — a Stop takes them
    // even without the play snapshot doing it (which is what the apps rely on).
    CHECK(grainMesh(scene) == nullptr);
    CHECK(granular.activeGrainCount() == 0);
    CHECK(granular.granularNodeCount() == 1);// kept for the status readout

    // Second episode, same document: a new solver, a new group, and no grains
    // carried over from the last one.
    physics.start(scene);
    granular.start(scene);
    REQUIRE_FALSE(granular.declined());
    CHECK(granular.groupCount() == 1);
    CHECK(granular.activeGrainCount() == 0);

    for (int i = 0; i < 60; ++i) {
        physics.update(kDt);
        granular.update(kDt);
    }
    const unsigned second = granular.activeGrainCount();
    CHECK(second > 0);
    // The emitter's own accumulator restarts too, so one second of the second
    // episode pours what one second of the first did.
    CHECK(second == afterOneSecond);

    granular.stop();
    physics.stop();
}


// No GPU tag: this is the path a machine WITHOUT one takes, and it has to work
// there — which is the whole point of a decline that is not a crash.
TEST_CASE("a granular chute with no physics declines and says so", "[physx]") {

    Scene scene;
    authorScene(scene);

    // No physics session at all: the same shape as a Play with the world
    // half of the build missing. The chute is still COUNTED.
    GranularPlaySession granular;
    std::string logged;
    granular.setLogger([&](const std::string& message) { logged = message; });

    granular.start(scene);
    CHECK(granular.granularNodeCount() == 1);
    CHECK(granular.groupCount() == 0);
    CHECK(granular.declined());
    CHECK(granular.reason() == logged);
    CHECK(logged.find("granular:") == 0);

    granular.update(kDt);// a no-op, not a crash
    CHECK(granular.activeGrainCount() == 0);
    granular.stop();
}
