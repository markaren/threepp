// Play runs on an in-memory copy of the scene (PlayCopy): nodes and materials
// copied with the same uuids, geometry, textures and clips shared, skeletons,
// light targets, LOD levels and robots rebuilt over the copied nodes, a splat
// cloud lent. Stop swaps back to the untouched authored scene.
//
// The exporter is deterministic, so it is the oracle: the copy must write the
// same text as the authored scene (fidelity), and the authored scene must
// write the same text before Play and after Stop (isolation). That keeps the
// serializer and the copier from drifting apart unnoticed.

#include <catch2/catch_test_macros.hpp>

#include "ExampleScenes.hpp"
#include "Scripting.hpp"

#include "threepp/extras/editor/ConveyorPlaySession.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/PlayCopy.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/splats/SplatData.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::string text(Object3D& scene) {

        ObjectExporter exporter;
        ObjectExporterOptions options;
        options.images = ImageStorage::Omit;
        options.models = ModelStorage::Embed;
        options.prettyPrint = false;
        return exporter.toJson(scene, options);
    }

    // Empty when the two documents are the same; otherwise both around the
    // first byte where they part, which is what a failure has to show (the
    // documents themselves run to megabytes).
    std::string firstDifference(const std::string& a, const std::string& b) {

        if (a == b) return {};
        std::size_t i = 0;
        while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
        const auto from = i > 160 ? i - 160 : 0;
        return "first: ..." + a.substr(from, 320) + " | second: ..." + b.substr(from, 320);
    }

    Object3D* byUuid(Object3D& root, const std::string& uuid) {

        Object3D* found = nullptr;
        root.traverse([&](Object3D& o) {
            if (!found && o.uuid == uuid) found = &o;
        });
        return found;
    }

    // Every node type the play copy handles that the shipped examples may lack.
    struct HandBuilt {

        std::shared_ptr<Scene> scene = Scene::create();
        std::shared_ptr<SkinnedMesh> skinned;
        std::shared_ptr<Mesh> first, second;
        std::shared_ptr<SplatCloud> cloud;
        std::shared_ptr<Robot> robot;

        HandBuilt() {

            scene->name = "Hand built";
            auto shared = MeshStandardMaterial::create();
            shared->color = Color(0x3366aa);
            auto box = BoxGeometry::create(1, 1, 1);

            first = Mesh::create(box, shared);
            first->name = "first";
            first->position.set(1, 2, 3);
            first->userData["note"] = std::string("kept");
            scene->add(first);
            second = Mesh::create(box, shared);
            second->name = "second";
            scene->add(second);

            // A skeleton whose bones are elsewhere in the scene.
            auto root = Bone::create();
            root->name = "root bone";
            auto tip = Bone::create();
            tip->name = "tip bone";
            tip->position.y = 1.f;
            root->add(tip);
            scene->add(root);
            scene->updateMatrixWorld(true);
            skinned = SkinnedMesh::create(BoxGeometry::create(0.5f, 2, 0.5f), MeshStandardMaterial::create());
            skinned->name = "skinned";
            scene->add(skinned);
            skinned->bind(Skeleton::create({root, tip}));

            auto instanced = InstancedMesh::create(box, MeshStandardMaterial::create(), 3);
            instanced->name = "instanced";
            for (int i = 0; i < 3; ++i) {
                instanced->setMatrixAt(i, Matrix4().makeTranslation(static_cast<float>(i), 0, 0));
            }
            scene->add(instanced);

            auto lod = LOD::create();
            lod->name = "lod";
            auto near = Mesh::create(box, MeshStandardMaterial::create());
            near->name = "near";
            auto far = Mesh::create(box, MeshStandardMaterial::create());
            far->name = "far";
            lod->addLevel(near, 0);
            lod->addLevel(far, 20);
            scene->add(lod);

            auto aim = Object3D::create();
            aim->name = "aim";
            aim->position.set(0, 0, -5);
            scene->add(aim);
            auto spot = SpotLight::create(Color(0xffffff), 2.f);
            spot->name = "spot";
            spot->setTarget(*aim);
            scene->add(spot);
            auto sun = DirectionalLight::create(Color(0xffeedd), 1.5f);
            sun->name = "sun";
            sun->castShadow = true;
            sun->setTarget(*aim);
            scene->add(sun);

            auto perspective = PerspectiveCamera::create(50, 1.5f, 0.1f, 100);
            perspective->name = "perspective";
            scene->add(perspective);
            auto ortho = OrthographicCamera::create(-2, 2, 1, -1, 0.1f, 50);
            ortho->name = "ortho";
            scene->add(ortho);

            auto sprite = Sprite::create(SpriteMaterial::create());
            sprite->name = "sprite";
            sprite->center.set(0.25f, 0.75f);
            scene->add(sprite);

            auto lineGeometry = BufferGeometry::create();
            lineGeometry->setAttribute("position", FloatBufferAttribute::create(std::vector<float>{0, 0, 0, 1, 1, 1}, 3));
            auto line = Line::create(lineGeometry, LineBasicMaterial::create());
            line->name = "line";
            scene->add(line);
            auto points = Points::create(lineGeometry, PointsMaterial::create());
            points->name = "points";
            scene->add(points);

            // A two-link robot on one revolute joint, posed off its rest.
            robot = Robot::create();
            robot->name = "robot";
            auto base = Group::create();
            base->name = "base";
            auto arm = Group::create();
            arm->name = "arm";
            auto joint = Object3D::create();
            joint->name = "shoulder";
            joint->position.set(0, 0.5f, 0);
            robot->addLink(base);
            robot->addLink(arm);
            Robot::JointInfo info;
            info.axis.set(0, 0, 1);
            info.type = Robot::JointType::Revolute;
            info.name = "shoulder";
            info.range = Robot::JointRange{-1.f, 1.f};
            info.parent = "base";
            info.child = "arm";
            robot->addJoint(joint, info);
            robot->finalize();
            robot->setJointValue(0, 0.3f);
            scene->add(robot);

            SplatGenerator::Options splats;
            splats.count = 50;
            splats.seed = 11u;
            splats.shDegree = 0;
            splats.includeDegenerates = false;
            cloud = SplatCloud::create(SplatGenerator::generate(splats));
            cloud->name = "cloud";
            cloud->position.set(4, 0, 0);
            cloud->setPointMix(0.25f);
            scene->add(cloud);

            // After the cloud, so its return to the same index is visible.
            auto last = Group::create();
            last->name = "last";
            scene->add(last);
        }
    };

    // The timber yard's rig from EditorExampleScene_test: physics, conveyors,
    // sensors and scripts, registered and stopped in the editor's order.
    struct YardRig {

        SceneDocument document;
        PhysicsPlaySession physics;
        ConveyorPlaySession conveyors;
        PhysxSensorPlaySession sensors;
        ScriptPlaySession scripts;

        YardRig() {
            conveyors.setPhysics(&physics);
            sensors.setPhysics(&physics);
        }

        void start() {
            physics.start(document.scene());
            conveyors.start(document.scene());
            sensors.start(document.scene());
            scripts.start(document.scene());
        }

        void run(int frames) {
            constexpr float dt = 1.f / 60.f;
            for (int i = 0; i < frames; ++i) {
                physics.update(dt);
                conveyors.update(dt);
                sensors.update(dt);
                scripts.update(dt);
            }
        }

        void stop() {
            scripts.stop();
            sensors.stop();
            conveyors.stop();
            physics.stop();
        }
    };

}// namespace


TEST_CASE("T1 fidelity: the play copy writes the same document as the authored scene", "[editor][playcopy]") {

    SECTION("every shipped example") {

        REQUIRE_FALSE(examples::all().empty());
        for (const auto& example : examples::all()) {
            INFO(std::string(example.slug));
            SceneDocument document;
            std::string error;
            REQUIRE(document.openJson(examples::json(example.slug), &error));

            const auto authored = text(document.scene());
            REQUIRE(document.beginPlay(&error));
            CHECK(firstDifference(authored, text(document.scene())).empty());
            document.endPlay();
        }
    }

    SECTION("a hand-built scene with every node type the copier rebuilds") {

        HandBuilt built;
        const auto authored = text(*built.scene);

        auto play = PlayCopy::build(*built.scene);
        REQUIRE(play);
        CHECK(firstDifference(authored, text(play->scene())).empty());
    }
}

TEST_CASE("T2 isolation: play and stop leave the authored scene as it was", "[editor][playcopy]") {

    YardRig rig;
    std::string error;
    const auto json = examples::json("timber-yard");
    REQUIRE_FALSE(json.empty());
    REQUIRE(rig.document.openJson(json, &error));

    auto* authored = &rig.document.scene();
    const auto before = text(*authored);

    REQUIRE(rig.document.beginPlay(&error));
    rig.start();
    rig.run(300);
    rig.stop();
    rig.document.endPlay();

    REQUIRE(&rig.document.scene() == authored);
    CHECK(rig.document.warnings().empty());
    CHECK(firstDifference(before, text(*authored)).empty());
}

TEST_CASE("T3 sharing: assets shared, instances and materials copied, uuids kept", "[editor][playcopy]") {

    HandBuilt built;
    const auto cloudIndex = static_cast<std::size_t>(
            std::find(built.scene->children.begin(), built.scene->children.end(), built.cloud.get()) -
            built.scene->children.begin());

    SceneDocument document;
    document.replaceScene(built.scene);
    REQUIRE(document.beginPlay());
    auto& copy = document.scene();
    REQUIRE(&copy != built.scene.get());

    // Meshes: same geometry, a different material with the same uuid, one
    // clone per distinct material.
    auto* first = dynamic_cast<Mesh*>(byUuid(copy, built.first->uuid));
    auto* second = dynamic_cast<Mesh*>(byUuid(copy, built.second->uuid));
    REQUIRE(first);
    REQUIRE(second);
    CHECK(first != built.first.get());
    CHECK(first->geometry() == built.first->geometry());
    CHECK(first->material() != built.first->material());
    CHECK(first->material()->uuid() == built.first->material()->uuid());
    CHECK(second->material() == first->material());

    // The skeleton is a new one over the COPIED bones, with the same uuid.
    auto* skinned = dynamic_cast<SkinnedMesh*>(byUuid(copy, built.skinned->uuid));
    REQUIRE(skinned);
    REQUIRE(skinned->skeleton);
    CHECK(skinned->skeleton != built.skinned->skeleton);
    CHECK(skinned->skeleton->uuid() == built.skinned->skeleton->uuid());
    REQUIRE(skinned->skeleton->bones.size() == built.skinned->skeleton->bones.size());
    for (std::size_t i = 0; i < skinned->skeleton->bones.size(); ++i) {
        const auto& bone = skinned->skeleton->bones[i];
        CHECK(bone.get() != built.skinned->skeleton->bones[i].get());
        CHECK(byUuid(copy, bone->uuid) == bone.get());
    }

    // The robot is live over the copied joint.
    auto* robot = dynamic_cast<Robot*>(byUuid(copy, built.robot->uuid));
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 1);
    CHECK(robot->jointNodes().front().get() == byUuid(copy, built.robot->jointNodes().front()->uuid));
    robot->setJointValue(0, -0.8f);
    CHECK(built.robot->getJointValue(0) == 0.3f);

    // The cloud is lent: the same object, in the copy, at the same index.
    CHECK(built.cloud->parent == &copy);
    CHECK(copy.children[cloudIndex] == built.cloud.get());
    built.cloud->position.set(-9, -9, -9);
    built.cloud->setPointMix(1.f);
    built.cloud->add(Object3D::create());

    document.endPlay();
    CHECK(&document.scene() == built.scene.get());
    CHECK(document.warnings().empty());

    // ...and handed back as it was authored.
    CHECK(built.cloud->parent == built.scene.get());
    CHECK(built.scene->children[cloudIndex] == built.cloud.get());
    CHECK(built.cloud->position.equals({4, 0, 0}));
    CHECK(built.cloud->pointMix() == 0.25f);
    CHECK(built.cloud->children.empty());
}

TEST_CASE("play copy timing on real models", "[.bench]") {

    // THREEPP_PLAYCOPY_BENCH_MODELS: model files, separated by ';'.
    const char* list = std::getenv("THREEPP_PLAYCOPY_BENCH_MODELS");
    if (!list || !*list) {
        WARN("THREEPP_PLAYCOPY_BENCH_MODELS is not set");
        return;
    }

    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point from) {
        return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
    };

    std::stringstream paths(list);
    std::string path;
    while (std::getline(paths, path, ';')) {
        if (path.empty()) continue;

        ModelLoader loader;
        auto model = loader.load(path);
        REQUIRE(model);

        SceneDocument document;
        document.scene().add(model);
        std::size_t nodes = 0;
        document.scene().traverse([&](Object3D&) { ++nodes; });

        const auto begin = Clock::now();
        REQUIRE(document.beginPlay());
        const auto beginMs = ms(begin);

        const auto end = Clock::now();
        document.endPlay();
        const auto endMs = ms(end);

        std::printf("%s (%zu nodes): beginPlay %.2f ms, endPlay %.2f ms\n",
                    path.c_str(), nodes, beginMs, endMs);
    }
}
