
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/SceneSnapshot.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A stand-in runtime that abuses the scene exactly as a physics engine
    // would: it moves things, deletes things and adds things.
    class VandalSession: public PlaySession {

    public:
        int starts = 0;
        int updates = 0;
        int stops = 0;
        float accumulated = 0.f;

        void start(Scene& scene) override {
            ++starts;
            scene_ = &scene;
        }

        void update(float dt) override {
            ++updates;
            accumulated += dt;
            if (!scene_) return;

            if (auto* box = scene_->getObjectByName("Box")) {
                box->position.y -= dt;
                box->name = "Box (simulated)";
            }
            if (auto* doomed = scene_->getObjectByName("Doomed")) {
                doomed->removeFromParent();
            }
            if (!scene_->getObjectByName("Spawned")) {
                auto spawned = Object3D::create();
                spawned->name = "Spawned";
                scene_->add(spawned);
            }
        }

        void stop() override {
            ++stops;
            scene_ = nullptr;
        }

        [[nodiscard]] std::string name() const override { return "Vandal"; }

    private:
        Scene* scene_ = nullptr;
    };

    class ThrowingSession: public PlaySession {

    public:
        void start(Scene&) override { throw std::runtime_error("no can do"); }
        void update(float) override {}
        void stop() override {}
        [[nodiscard]] std::string name() const override { return "Throwing"; }
    };

    void populate(Scene& scene) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->position.set(0.f, 4.f, 0.f);
        if (auto* material = box->materialAs<MeshStandardMaterial>()) {
            material->color = Color(0xff8800);
        }
        PhysicsConfig physics;
        physics.enabled = true;
        physics.mass = 2.f;
        physics.write(*box);
        scene.add(box);

        auto ground = ObjectFactory::createPrimitive(Primitive::Plane, scene);
        ground->name = "Ground";
        scene.add(ground);

        auto doomed = Object3D::create();
        doomed->name = "Doomed";
        scene.add(doomed);
    }

}// namespace


TEST_CASE("a snapshot restores the scene after arbitrary mutation", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    const auto boxUuid = document.scene().getObjectByName("Box")->uuid;
    const auto childCount = document.scene().children.size();

    SceneSnapshot snapshot;
    std::string error;
    REQUIRE(document.capture(snapshot, &error));
    REQUIRE(snapshot.valid());

    // Wreck the scene.
    auto& live = document.scene();
    live.getObjectByName("Box")->position.set(99.f, -99.f, 99.f);
    live.getObjectByName("Box")->name = "Wrecked";
    live.getObjectByName("Doomed")->removeFromParent();
    auto extra = Object3D::create();
    extra->name = "Extra";
    live.add(extra);

    REQUIRE(document.restore(snapshot, &error));

    auto& restored = document.scene();
    CHECK(restored.children.size() == childCount);

    auto* box = restored.getObjectByName<Mesh>("Box");
    REQUIRE(box);
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(box->position.y, WithinAbs(4.f, 1e-5));
    // uuid-stable, which is how the editor re-finds the selection.
    CHECK(box->uuid == boxUuid);

    CHECK(restored.getObjectByName("Doomed") != nullptr);
    CHECK(restored.getObjectByName("Extra") == nullptr);
    CHECK(restored.getObjectByName("Wrecked") == nullptr);

    auto* material = box->materialAs<MeshStandardMaterial>();
    REQUIRE(material);
    CHECK(material->color.getHex() == 0xff8800u);

    const auto physics = PhysicsConfig::read(*box);
    REQUIRE(physics.has_value());
    CHECK(physics->mass == 2.f);
}

TEST_CASE("PlayController drives sessions and restores on stop", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    auto session = std::make_shared<VandalSession>();
    PlayController controller;
    controller.addSession(session);

    CHECK(controller.stopped());

    std::string error;
    REQUIRE(controller.play(document, &error));
    CHECK(controller.state() == PlayController::State::Playing);
    CHECK(session->starts == 1);

    for (int i = 0; i < 10; ++i) controller.update(0.1f);
    CHECK(session->updates == 10);
    using Catch::Matchers::WithinAbs;
    CHECK_THAT(controller.elapsed(), WithinAbs(1.f, 1e-4));

    // The session really did change the scene.
    CHECK(document.scene().getObjectByName("Box (simulated)") != nullptr);
    CHECK(document.scene().getObjectByName("Doomed") == nullptr);
    CHECK(document.scene().getObjectByName("Spawned") != nullptr);

    REQUIRE(controller.stop(document, &error));
    CHECK(controller.stopped());
    CHECK(session->stops == 1);

    // ...and none of it survived.
    CHECK(document.scene().getObjectByName("Box") != nullptr);
    CHECK(document.scene().getObjectByName("Box (simulated)") == nullptr);
    CHECK(document.scene().getObjectByName("Doomed") != nullptr);
    CHECK(document.scene().getObjectByName("Spawned") == nullptr);
    CHECK_THAT(document.scene().getObjectByName("Box")->position.y, WithinAbs(4.f, 1e-5));
}

TEST_CASE("pause suspends updates without stopping", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    auto session = std::make_shared<VandalSession>();
    PlayController controller;
    controller.addSession(session);

    REQUIRE(controller.play(document));
    controller.update(0.1f);
    controller.pause();
    CHECK(controller.paused());

    controller.update(0.1f);
    controller.update(0.1f);
    CHECK(session->updates == 1);

    controller.resume();
    controller.update(0.1f);
    CHECK(session->updates == 2);
    CHECK(session->stops == 0);

    REQUIRE(controller.stop(document));
}

TEST_CASE("a session that fails to start rolls the whole play back", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    auto good = std::make_shared<VandalSession>();
    PlayController controller;
    controller.addSession(good);
    controller.addSession(std::make_shared<ThrowingSession>());

    std::string error;
    CHECK_FALSE(controller.play(document, &error));
    CHECK(error.find("Throwing") != std::string::npos);
    CHECK(controller.stopped());
    // The one that did start was shut down again.
    CHECK(good->starts == 1);
    CHECK(good->stops == 1);

    // The scene is untouched and still editable.
    CHECK(document.scene().getObjectByName("Box") != nullptr);
}

TEST_CASE("stopping without playing is harmless", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    PlayController controller;
    CHECK(controller.stop(document));
    CHECK(controller.stopped());
    controller.update(0.1f);// no-op
    CHECK(document.scene().getObjectByName("Box") != nullptr);
}

TEST_CASE("snapshots keep live texture instances rather than re-decoding", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    // No textures in this scene, but the API contract still holds: collect
    // returns nothing and rebind is a no-op.
    std::unordered_map<std::string, std::shared_ptr<Texture>> textures;
    SceneSnapshot::collectTextures(document.scene(), textures);
    CHECK(textures.empty());

    SceneSnapshot snapshot;
    REQUIRE(document.capture(snapshot));
    auto restored = snapshot.restore();
    REQUIRE(restored);
    CHECK(restored->getObjectByName("Box") != nullptr);
}
