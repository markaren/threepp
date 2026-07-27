
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <filesystem>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A scene of the shape the editor actually produces: primitives with
    // standard materials, a light, a nested group, and physics authored into
    // userData.
    void populate(Scene& scene) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->position.set(1.f, 0.5f, -2.f);
        box->rotation.y = 0.5f;
        box->scale.set(2.f, 1.f, 0.5f);
        box->castShadow = true;
        box->renderOrder = -3;
        if (auto* material = box->materialAs<MeshStandardMaterial>()) {
            material->color = Color(0x3366cc);
            material->roughness = 0.25f;
            material->metalness = 0.75f;
            material->opacity = 0.5f;
            material->transparent = true;
        }

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Convex;
        physics.mass = 12.5f;
        physics.friction = 0.8f;
        physics.restitution = 0.35f;
        physics.write(*box);

        auto group = ObjectFactory::createGroup(scene);
        group->position.set(0.f, 3.f, 0.f);

        auto sphere = ObjectFactory::createPrimitive(Primitive::Sphere, scene);
        sphere->name = "Child Sphere";
        group->add(sphere);

        auto light = ObjectFactory::createLight(LightKind::Directional, scene);

        scene.add(box);
        scene.add(group);
        scene.add(light);
    }

    std::filesystem::path scratchDir() {

        auto directory = std::filesystem::temp_directory_path() / "threepp_editor_tests";
        std::filesystem::create_directories(directory);
        return directory;
    }

}// namespace


TEST_CASE("PhysicsConfig round-trips through userData", "[editor]") {

    auto object = Object3D::create();

    PhysicsConfig config;
    config.enabled = true;
    config.body = PhysicsConfig::Body::Kinematic;
    config.shape = PhysicsConfig::Shape::Capsule;
    config.mass = 3.25f;
    config.friction = 0.9f;
    config.restitution = 0.125f;
    config.write(*object);

    const auto read = PhysicsConfig::read(*object);
    REQUIRE(read.has_value());
    CHECK(read->enabled);
    CHECK(read->body == PhysicsConfig::Body::Kinematic);
    CHECK(read->shape == PhysicsConfig::Shape::Capsule);
    CHECK(read->mass == 3.25f);
    CHECK(read->friction == 0.9f);
    CHECK(read->restitution == 0.125f);

    // The encoding is deterministic, so an unchanged scene produces an
    // unchanged document.
    CHECK(config.encode() == read->encode());

    // Disabling clears the entry entirely rather than writing enabled=false.
    PhysicsConfig off;
    off.enabled = false;
    off.write(*object);
    CHECK_FALSE(PhysicsConfig::read(*object).has_value());
}

TEST_CASE("PhysicsConfig tolerates unknown and malformed input", "[editor]") {

    // A document written by a newer editor.
    const auto forward = PhysicsConfig::decode("body=static;shape=box;mass=2;newKey=whatever");
    REQUIRE(forward.has_value());
    CHECK(forward->body == PhysicsConfig::Body::Static);
    CHECK(forward->mass == 2.f);

    // Garbage values fall back to the defaults rather than throwing.
    const auto broken = PhysicsConfig::decode("body=;shape=;mass=abc");
    REQUIRE(broken.has_value());
    CHECK(broken->body == PhysicsConfig::Body::Dynamic);
    CHECK(broken->shape == PhysicsConfig::Shape::Auto);
    CHECK(broken->mass == 1.f);
}

TEST_CASE("an editor scene round-trips through the document format", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    const auto json = document.toJson(false);
    REQUIRE_FALSE(json.empty());

    ObjectLoader loader;
    auto parsed = loader.parse(json);
    REQUIRE(parsed);
    auto* reloaded = dynamic_cast<Scene*>(parsed.get());
    REQUIRE(reloaded);

    auto* box = reloaded->getObjectByName<Mesh>("Box");
    REQUIRE(box);

    using Catch::Matchers::WithinAbs;
    CHECK_THAT(box->position.x, WithinAbs(1.f, 1e-5));
    CHECK_THAT(box->position.y, WithinAbs(0.5f, 1e-5));
    CHECK_THAT(box->position.z, WithinAbs(-2.f, 1e-5));
    CHECK_THAT(box->scale.x, WithinAbs(2.f, 1e-5));
    CHECK(box->castShadow);
    CHECK(box->renderOrder == -3);

    auto* material = box->materialAs<MeshStandardMaterial>();
    REQUIRE(material);
    CHECK_THAT(material->roughness, WithinAbs(0.25f, 1e-5));
    CHECK_THAT(material->metalness, WithinAbs(0.75f, 1e-5));
    CHECK_THAT(material->opacity, WithinAbs(0.5f, 1e-5));
    CHECK(material->transparent);
    CHECK(material->color.getHex() == 0x3366ccu);

    const auto physics = PhysicsConfig::read(*box);
    REQUIRE(physics.has_value());
    CHECK(physics->body == PhysicsConfig::Body::Dynamic);
    CHECK(physics->shape == PhysicsConfig::Shape::Convex);
    CHECK_THAT(physics->mass, WithinAbs(12.5f, 1e-5));
    CHECK_THAT(physics->friction, WithinAbs(0.8f, 1e-5));
    CHECK_THAT(physics->restitution, WithinAbs(0.35f, 1e-5));

    // Hierarchy survives.
    auto* sphere = reloaded->getObjectByName<Mesh>("Child Sphere");
    REQUIRE(sphere);
    REQUIRE(sphere->parent);
    CHECK(sphere->parent->name == "Group");

    // uuids are adopted verbatim, which is what lets the editor re-resolve a
    // selection after a reload.
    CHECK(box->uuid == document.scene().getObjectByName("Box")->uuid);
}

TEST_CASE("the document is deterministic", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    const auto first = document.toJson(false);
    const auto second = document.toJson(false);
    CHECK(first == second);
}

TEST_CASE("editor-only objects are excluded from the document", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    auto overlay = Group::create();
    overlay->name = "__editor_overlay";
    auto helper = Group::create();
    helper->name = "__editor_grid";
    overlay->add(helper);

    document.addEditorOnly(*overlay);

    // It IS in the scene (so it renders)...
    CHECK(overlay->parent == &document.scene());
    CHECK(document.isEditorOnly(*overlay));
    CHECK(document.isEditorOnly(*helper));

    // ...and is not in the document.
    const auto json = document.toJson(false);
    CHECK(json.find("__editor_overlay") == std::string::npos);
    CHECK(json.find("__editor_grid") == std::string::npos);

    // ...and is put back afterwards.
    CHECK(overlay->parent == &document.scene());
    CHECK(document.scene().getObjectByName("Box") != nullptr);
}

TEST_CASE("save and open round-trip a scene through a file", "[editor]") {

    const auto path = scratchDir() / "round_trip.json";
    std::filesystem::remove(path);

    SceneDocument document;
    populate(document.scene());
    document.setDirty(true);

    std::string error;
    REQUIRE(document.saveAs(path, &error));
    CHECK(error.empty());
    CHECK_FALSE(document.dirty());
    CHECK(document.path() == path);
    CHECK(document.displayName() == "round_trip.json");
    CHECK(document.title() == "round_trip.json");
    REQUIRE(std::filesystem::exists(path));

    SceneDocument reopened;
    REQUIRE(reopened.open(path, &error));
    CHECK_FALSE(reopened.dirty());

    auto* box = reopened.scene().getObjectByName<Mesh>("Box");
    REQUIRE(box);
    CHECK(PhysicsConfig::read(*box).has_value());

    std::filesystem::remove(path);
}

TEST_CASE("a dirty document is marked in its title", "[editor]") {

    SceneDocument document;
    CHECK(document.displayName() == "untitled");
    CHECK(document.title() == "untitled");

    document.setDirty(true);
    CHECK(document.title() == "untitled*");
}

TEST_CASE("newScene replaces the scene and notifies", "[editor]") {

    SceneDocument document;
    populate(document.scene());

    Scene* notified = nullptr;
    document.onSceneReplaced([&](Scene& scene) { notified = &scene; });

    auto* previous = &document.scene();
    document.newScene();

    CHECK(&document.scene() != previous);
    CHECK(notified == &document.scene());
    CHECK(document.scene().children.empty());
    CHECK_FALSE(document.dirty());
}
