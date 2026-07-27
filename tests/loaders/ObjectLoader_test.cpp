
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/Texture.hpp"

#include <any>
#include <string>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    template<class T = Object3D>
    T* findByUuid(Object3D& root, const std::string& uuid) {

        T* found = nullptr;
        root.traverse([&](Object3D& o) {
            if (!found && o.uuid == uuid) found = dynamic_cast<T*>(&o);
        });
        return found;
    }

    std::shared_ptr<BufferGeometry> makeDataGeometry() {

        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(
                                                   std::vector<float>{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, 3));
        geometry->setAttribute("uv", FloatBufferAttribute::create(
                                             std::vector<float>{0, 0, 1, 0, 1, 1, 0, 1}, 2));
        geometry->setIndex(std::vector<unsigned int>{0, 1, 2, 2, 3, 0});
        geometry->addGroup(0, 3, 0);
        geometry->addGroup(3, 3, 1);
        geometry->computeBoundingSphere();

        return geometry;
    }

}// namespace


TEST_CASE("Object JSON round-trip preserves a mixed scene") {

    auto scene = Scene::create();
    scene->name = "root";
    scene->background = Background(Color(0x203040));
    scene->fog = Fog(Color(0x445566), 5.f, 250.f);
    scene->userData["label"] = std::string("demo");
    scene->userData["count"] = 7;
    scene->userData["enabled"] = true;
    scene->userData["scale"] = 1.5;

    auto group = Group::create();
    group->name = "group";
    group->position.set(1, 2, 3);
    group->scale.set(2, 2, 2);
    group->renderOrder = 3;
    scene->add(group);

    auto boxMaterial = MeshStandardMaterial::create();
    boxMaterial->color = Color(0xff8800);
    boxMaterial->roughness = 0.375f;
    boxMaterial->metalness = 0.625f;
    boxMaterial->name = "boxMaterial";

    auto box = Mesh::create(BoxGeometry::create(2, 3, 4, 2, 1, 3), boxMaterial);
    box->name = "box";
    box->castShadow = true;
    box->position.set(-1, 0.5f, 2);
    group->add(box);

    auto dataMesh = Mesh::create(makeDataGeometry(), MeshStandardMaterial::create());
    dataMesh->name = "dataMesh";
    scene->add(dataMesh);

    auto instanced = InstancedMesh::create(BoxGeometry::create(), MeshStandardMaterial::create(), 3);
    instanced->name = "instanced";
    for (unsigned i = 0; i < 3; ++i) {
        Matrix4 m;
        m.setPosition(static_cast<float>(i), static_cast<float>(i) * 2, 0);
        instanced->setMatrixAt(i, m);
        instanced->setColorAt(i, Color(0.25f * static_cast<float>(i), 0.5f, 1.f));
    }
    scene->add(instanced);

    // skinned mesh + skeleton
    auto boneA = Bone::create();
    boneA->name = "boneA";
    auto boneB = Bone::create();
    boneB->name = "boneB";
    boneB->position.set(0, 1, 0);
    boneA->add(boneB);

    auto skinned = SkinnedMesh::create(makeDataGeometry(), MeshStandardMaterial::create());
    skinned->name = "skinned";
    skinned->add(boneA);
    auto skeleton = Skeleton::create({boneA, boneB});
    skinned->bind(skeleton);
    scene->add(skinned);

    auto line = Line::create(makeDataGeometry(), LineBasicMaterial::create());
    line->name = "line";
    scene->add(line);

    auto points = Points::create(makeDataGeometry(), PointsMaterial::create());
    points->name = "points";
    scene->add(points);

    auto sprite = Sprite::create(SpriteMaterial::create());
    sprite->name = "sprite";
    scene->add(sprite);

    auto lod = LOD::create();
    lod->name = "lod";
    auto lodNear = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
    auto lodFar = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
    lod->addLevel(lodNear, 0.f);
    lod->addLevel(lodFar, 25.f);
    scene->add(lod);

    auto camera = PerspectiveCamera::create(42.5f, 1.75f, 0.25f, 900.f);
    camera->name = "camera";
    camera->zoom = 1.25f;
    camera->filmGauge = 24.f;
    camera->filmOffset = 1.5f;
    scene->add(camera);

    auto dirLight = DirectionalLight::create(Color(0xffeecc), 0.85f);
    dirLight->name = "sun";
    dirLight->castShadow = true;
    dirLight->shadow->bias = -0.0015f;
    dirLight->shadow->radius = 3.f;
    dirLight->shadow->mapSize.set(1024, 512);
    scene->add(dirLight);

    auto ambient = AmbientLight::create(Color(0x223344), 0.4f);
    ambient->name = "ambient";
    scene->add(ambient);

    auto clip = std::make_shared<AnimationClip>(
            "spin", 2.f,
            std::vector<std::shared_ptr<KeyframeTrack>>{
                    std::make_shared<VectorKeyframeTrack>(
                            "group.position", std::vector<float>{0.f, 1.f, 2.f},
                            std::vector<float>{0, 0, 0, 1, 2, 3, 0, 4, 0}),
                    std::make_shared<QuaternionKeyframeTrack>(
                            "group.quaternion", std::vector<float>{0.f, 2.f},
                            std::vector<float>{0, 0, 0, 1, 0, 1, 0, 0})});
    scene->animations.push_back(clip);

    // capture pre-export identities
    const auto sceneUuid = scene->uuid;
    const auto boxUuid = box->uuid;
    const auto boxMaterialUuid = boxMaterial->uuid();
    const auto dataGeometryUuid = dataMesh->geometry()->uuid;
    const auto skeletonUuid = skeleton->uuid();
    const auto lodNearUuid = lodNear->uuid;
    const auto lodFarUuid = lodFar->uuid;

    ObjectExporter exporter;
    const auto text = exporter.toJson(*scene);

    ObjectLoader loader;
    auto parsed = loader.parse(text);
    REQUIRE(parsed != nullptr);

    SECTION("scene identity, fog, background and userData") {

        auto* parsedScene = parsed->as<Scene>();
        REQUIRE(parsedScene != nullptr);
        CHECK(parsedScene->uuid == sceneUuid);
        CHECK(parsedScene->name == "root");

        REQUIRE(parsedScene->background.isColor());
        CHECK(parsedScene->background.color().getHex() == 0x203040);

        REQUIRE(parsedScene->fog.has_value());
        REQUIRE(std::holds_alternative<Fog>(*parsedScene->fog));
        const auto& fog = std::get<Fog>(*parsedScene->fog);
        CHECK(fog.color.getHex() == 0x445566);
        CHECK_THAT(fog.nearPlane, WithinAbs(5.f, 1e-5));
        CHECK_THAT(fog.farPlane, WithinAbs(250.f, 1e-5));

        CHECK(std::any_cast<std::string>(parsedScene->userData.at("label")) == "demo");
        CHECK(std::any_cast<int>(parsedScene->userData.at("count")) == 7);
        CHECK(std::any_cast<bool>(parsedScene->userData.at("enabled")) == true);
        CHECK_THAT(std::any_cast<double>(parsedScene->userData.at("scale")), WithinAbs(1.5, 1e-9));
    }

    SECTION("transforms decompose back to position/quaternion/scale") {

        auto* parsedGroup = parsed->getObjectByName("group");
        REQUIRE(parsedGroup != nullptr);
        CHECK_THAT(parsedGroup->position.x, WithinAbs(1.f, 1e-5));
        CHECK_THAT(parsedGroup->position.y, WithinAbs(2.f, 1e-5));
        CHECK_THAT(parsedGroup->position.z, WithinAbs(3.f, 1e-5));
        CHECK_THAT(parsedGroup->scale.x, WithinAbs(2.f, 1e-5));
        CHECK(parsedGroup->renderOrder == 3);

        auto* parsedBox = findByUuid<Mesh>(*parsed, boxUuid);
        REQUIRE(parsedBox != nullptr);
        CHECK(parsedBox->castShadow);
        CHECK_THAT(parsedBox->position.x, WithinAbs(-1.f, 1e-5));
        CHECK_THAT(parsedBox->position.y, WithinAbs(0.5f, 1e-5));
        CHECK_THAT(parsedBox->position.z, WithinAbs(2.f, 1e-5));
    }

    SECTION("parametric geometry and material parameters") {

        auto* parsedBox = findByUuid<Mesh>(*parsed, boxUuid);
        REQUIRE(parsedBox != nullptr);

        auto geometry = std::dynamic_pointer_cast<BoxGeometry>(parsedBox->geometry());
        REQUIRE(geometry != nullptr);
        CHECK(geometry->type() == "BoxGeometry");
        CHECK_THAT(geometry->width, WithinAbs(2.f, 1e-5));
        CHECK_THAT(geometry->height, WithinAbs(3.f, 1e-5));
        CHECK_THAT(geometry->depth, WithinAbs(4.f, 1e-5));
        CHECK(geometry->parameters.widthSegments == 2);
        CHECK(geometry->parameters.depthSegments == 3);

        auto* material = parsedBox->materialAs<MeshStandardMaterial>();
        REQUIRE(material != nullptr);
        CHECK(material->uuid() == boxMaterialUuid);
        CHECK(material->name == "boxMaterial");
        CHECK(material->color.getHex() == 0xff8800);
        CHECK_THAT(material->roughness, WithinAbs(0.375f, 1e-6));
        CHECK_THAT(material->metalness, WithinAbs(0.625f, 1e-6));
    }

    SECTION("data-form geometry arrays are byte-equal") {

        auto* parsedMesh = parsed->getObjectByName("dataMesh");
        REQUIRE(parsedMesh != nullptr);

        auto geometry = parsedMesh->geometry();
        REQUIRE(geometry != nullptr);
        CHECK(geometry->uuid == dataGeometryUuid);

        const auto* position = geometry->getAttribute<float>("position");
        REQUIRE(position != nullptr);
        CHECK(position->itemSize() == 3);
        CHECK(position->array() == std::vector<float>{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0});

        const auto* uv = geometry->getAttribute<float>("uv");
        REQUIRE(uv != nullptr);
        CHECK(uv->array() == std::vector<float>{0, 0, 1, 0, 1, 1, 0, 1});

        REQUIRE(geometry->hasIndex());
        CHECK(geometry->getIndex()->array() == std::vector<unsigned int>{0, 1, 2, 2, 3, 0});

        REQUIRE(geometry->groups.size() == 2);
        CHECK(geometry->groups[1].start == 3);
        CHECK(geometry->groups[1].count == 3);
        CHECK(geometry->groups[1].materialIndex == 1);

        REQUIRE(geometry->boundingSphere.has_value());
        CHECK(geometry->boundingSphere->radius > 0.f);
    }

    SECTION("instanced matrices and colors") {

        auto* parsedInstanced = dynamic_cast<InstancedMesh*>(parsed->getObjectByName("instanced"));
        REQUIRE(parsedInstanced != nullptr);
        CHECK(parsedInstanced->count() == 3);

        Matrix4 m;
        parsedInstanced->getMatrixAt(2, m);
        CHECK_THAT(m.elements[12], WithinAbs(2.f, 1e-5));
        CHECK_THAT(m.elements[13], WithinAbs(4.f, 1e-5));

        REQUIRE(parsedInstanced->instanceColor() != nullptr);
        Color c;
        parsedInstanced->getColorAt(1, c);
        CHECK_THAT(c.r, WithinAbs(0.25f, 1e-5));
        CHECK_THAT(c.g, WithinAbs(0.5f, 1e-5));
        CHECK_THAT(c.b, WithinAbs(1.f, 1e-5));
    }

    SECTION("every resource keeps its uuid") {

        // An editor keys selection and undo off uuids, so identity must survive
        // for objects AND for every shared resource map.
        auto* parsedBox = findByUuid<Mesh>(*parsed, boxUuid);
        REQUIRE(parsedBox != nullptr);
        CHECK(parsedBox->geometry()->uuid == box->geometry()->uuid);
        CHECK(parsedBox->material()->uuid() == boxMaterialUuid);

        CHECK(parsed->getObjectByName("dataMesh")->geometry()->uuid == dataGeometryUuid);
        CHECK(findByUuid(*parsed, lodNearUuid) != nullptr);
        CHECK(findByUuid(*parsed, lodFarUuid) != nullptr);
        CHECK(findByUuid(*parsed, boneA->uuid) != nullptr);
        CHECK(findByUuid(*parsed, boneB->uuid) != nullptr);

        auto* parsedSkinned = dynamic_cast<SkinnedMesh*>(parsed->getObjectByName("skinned"));
        REQUIRE(parsedSkinned != nullptr);
        CHECK(parsedSkinned->skeleton->uuid() == skeletonUuid);

        REQUIRE(parsed->animations.size() == 1);
        CHECK(parsed->animations.front()->uuid() == clip->uuid());

        auto* sun = dynamic_cast<DirectionalLight*>(parsed->getObjectByName("sun"));
        REQUIRE(sun != nullptr);
        CHECK(sun->shadow->camera->uuid == dirLight->shadow->camera->uuid);
    }

    SECTION("export is deterministic") {

        ObjectExporter first;
        ObjectExporter second;
        CHECK(first.toJson(*scene) == second.toJson(*scene));
        CHECK(first.toJson(*scene) == text);
        CHECK(first.warnings().empty());
    }

    SECTION("skeleton rebinds to the parsed bones") {

        auto* parsedSkinned = dynamic_cast<SkinnedMesh*>(parsed->getObjectByName("skinned"));
        REQUIRE(parsedSkinned != nullptr);
        REQUIRE(parsedSkinned->skeleton != nullptr);
        CHECK(parsedSkinned->skeleton->uuid() == skeletonUuid);
        REQUIRE(parsedSkinned->skeleton->bones.size() == 2);
        CHECK(parsedSkinned->skeleton->bones[0]->name == "boneA");
        CHECK(parsedSkinned->skeleton->bones[1]->name == "boneB");
        CHECK(parsedSkinned->bindMode == SkinnedMesh::BindMode::Attached);
    }

    SECTION("LOD levels keep their objects and distances") {

        auto* parsedLod = dynamic_cast<LOD*>(parsed->getObjectByName("lod"));
        REQUIRE(parsedLod != nullptr);
        REQUIRE(parsedLod->getLevels().size() == 2);
        CHECK(parsedLod->getLevels()[0].object->uuid == lodNearUuid);
        CHECK_THAT(parsedLod->getLevels()[0].distance, WithinAbs(0.f, 1e-6));
        CHECK(parsedLod->getLevels()[1].object->uuid == lodFarUuid);
        CHECK_THAT(parsedLod->getLevels()[1].distance, WithinAbs(25.f, 1e-6));
        CHECK(parsedLod->children.size() == 2);
    }

    SECTION("camera and light parameters") {

        auto* parsedCamera = dynamic_cast<PerspectiveCamera*>(parsed->getObjectByName("camera"));
        REQUIRE(parsedCamera != nullptr);
        CHECK_THAT(parsedCamera->fov, WithinAbs(42.5f, 1e-5));
        CHECK_THAT(parsedCamera->aspect, WithinAbs(1.75f, 1e-5));
        CHECK_THAT(parsedCamera->nearPlane, WithinAbs(0.25f, 1e-5));
        CHECK_THAT(parsedCamera->farPlane, WithinAbs(900.f, 1e-3));
        CHECK_THAT(parsedCamera->zoom, WithinAbs(1.25f, 1e-5));
        CHECK_THAT(parsedCamera->filmGauge, WithinAbs(24.f, 1e-5));
        CHECK_THAT(parsedCamera->filmOffset, WithinAbs(1.5f, 1e-5));

        auto* sun = dynamic_cast<DirectionalLight*>(parsed->getObjectByName("sun"));
        REQUIRE(sun != nullptr);
        CHECK(sun->color.getHex() == 0xffeecc);
        CHECK_THAT(sun->intensity, WithinAbs(0.85f, 1e-5));
        CHECK(sun->castShadow);
        CHECK_THAT(sun->shadow->bias, WithinAbs(-0.0015f, 1e-7));
        CHECK_THAT(sun->shadow->radius, WithinAbs(3.f, 1e-6));
        CHECK_THAT(sun->shadow->mapSize.x, WithinAbs(1024.f, 1e-6));
        CHECK_THAT(sun->shadow->mapSize.y, WithinAbs(512.f, 1e-6));

        auto* parsedAmbient = dynamic_cast<AmbientLight*>(parsed->getObjectByName("ambient"));
        REQUIRE(parsedAmbient != nullptr);
        CHECK(parsedAmbient->color.getHex() == 0x223344);
        CHECK_THAT(parsedAmbient->intensity, WithinAbs(0.4f, 1e-5));
    }

    SECTION("line, points and sprite survive") {

        CHECK(dynamic_cast<Line*>(parsed->getObjectByName("line")) != nullptr);
        CHECK(dynamic_cast<Points*>(parsed->getObjectByName("points")) != nullptr);
        auto* parsedSprite = dynamic_cast<Sprite*>(parsed->getObjectByName("sprite"));
        REQUIRE(parsedSprite != nullptr);
        CHECK(parsedSprite->material() != nullptr);
    }

    SECTION("animation clip tracks are value-equal") {

        REQUIRE(parsed->animations.size() == 1);
        const auto& parsedClip = parsed->animations.front();
        CHECK(parsedClip->uuid() == clip->uuid());
        CHECK(parsedClip->name() == "spin");
        CHECK_THAT(parsedClip->getDuration(), WithinAbs(2.f, 1e-6));

        REQUIRE(parsedClip->getTracks().size() == 2);

        const auto& vec = parsedClip->getTracks()[0];
        CHECK(vec->ValueTypeName() == "vector");
        CHECK(vec->getName() == "group.position");
        CHECK(vec->getTimes() == std::vector<float>{0.f, 1.f, 2.f});
        CHECK(vec->getValues() == std::vector<float>{0, 0, 0, 1, 2, 3, 0, 4, 0});

        const auto& quat = parsedClip->getTracks()[1];
        CHECK(quat->ValueTypeName() == "quaternion");
        CHECK(quat->getTimes() == std::vector<float>{0.f, 2.f});
        CHECK(quat->getValues() == std::vector<float>{0, 0, 0, 1, 0, 1, 0, 0});
    }
}


TEST_CASE("A subtree exports on its own with just the resources it references") {

    auto scene = Scene::create();

    auto keptMaterial = MeshStandardMaterial::create();
    keptMaterial->color = Color(0x00ff00);

    auto group = Group::create();
    group->name = "subtree";
    group->position.set(4, 5, 6);
    auto child = Mesh::create(BoxGeometry::create(1, 2, 3), keptMaterial);
    child->name = "child";
    group->add(child);
    scene->add(group);

    // a sibling whose resources must NOT end up in the subtree document
    auto siblingMaterial = MeshStandardMaterial::create();
    auto sibling = Mesh::create(SphereGeometry::create(), siblingMaterial);
    scene->add(sibling);

    ObjectExporter exporter;
    const auto text = exporter.toJson(*group);

    CHECK(text.find(siblingMaterial->uuid()) == std::string::npos);
    CHECK(text.find(sibling->geometry()->uuid) == std::string::npos);

    ObjectLoader loader;
    auto parsed = loader.parse(text);
    REQUIRE(parsed != nullptr);
    CHECK(parsed->uuid == group->uuid);
    CHECK(parsed->name == "subtree");
    CHECK(parsed->parent == nullptr);
    CHECK_THAT(parsed->position.x, WithinAbs(4.f, 1e-5));

    REQUIRE(parsed->children.size() == 1);
    auto* parsedChild = parsed->children.front();
    CHECK(parsedChild->uuid == child->uuid);
    CHECK(parsedChild->geometry()->uuid == child->geometry()->uuid);
    CHECK(parsedChild->material()->uuid() == keptMaterial->uuid());
    CHECK(parsedChild->materialAs<MeshStandardMaterial>()->color.getHex() == 0x00ff00);
}


TEST_CASE("Lossy cases are reported through warnings()") {

    SECTION("export reports userData it cannot represent and imageless textures") {

        auto mesh = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
        mesh->userData["ok"] = 3;
        mesh->userData["nope"] = Vector3(1, 2, 3);
        mesh->materialAs<MeshStandardMaterial>()->map = Texture::create();

        ObjectExporter exporter;
        const auto text = exporter.toJson(*mesh);

        REQUIRE(exporter.warnings().size() == 2);
        const auto joined = exporter.warnings()[0] + "|" + exporter.warnings()[1];
        CHECK(joined.find("nope") != std::string::npos);
        CHECK(joined.find("no CPU-side image data") != std::string::npos);

        // a clean export leaves the list empty
        auto clean = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
        CHECK(exporter.toJson(*clean).size() > 0);
        CHECK(exporter.warnings().empty());
    }

    SECTION("import reports unknown types and dangling references") {

        const std::string text = R"({
            "metadata": { "version": 4.5, "type": "Object" },
            "object": {
                "uuid": "ROOT", "type": "Mesh", "layers": 1,
                "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                "geometry": "MISSING", "material": "ALSO-MISSING",
                "children": [
                    { "uuid": "X", "type": "NotAThing", "layers": 1,
                      "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] }
                ]
            }
        })";

        ObjectLoader loader;
        auto parsed = loader.parse(text);
        REQUIRE(parsed != nullptr);
        REQUIRE(loader.warnings().size() == 3);

        std::string joined;
        for (const auto& w : loader.warnings()) joined += w + "|";
        CHECK(joined.find("undefined geometry 'MISSING'") != std::string::npos);
        CHECK(joined.find("undefined material 'ALSO-MISSING'") != std::string::npos);
        CHECK(joined.find("unsupported object type 'NotAThing'") != std::string::npos);
    }

    SECTION("malformed input is reported rather than thrown") {

        ObjectLoader loader;
        CHECK(loader.parse("<not json>") == nullptr);
        REQUIRE(loader.warnings().size() == 1);
        CHECK(loader.warnings().front() == "malformed JSON");
    }
}


TEST_CASE("Texture round-trip with embedded images") {

    // 2x2 RGBA, four distinct byte-exact pixels
    const std::vector<unsigned char> pixels{
            255, 0, 0, 255, /**/ 0, 255, 0, 128,
            0, 0, 255, 255, /**/ 17, 34, 51, 68};

    auto texture = DataTexture::create(pixels, 2, 2);
    texture->name = "swatch";
    texture->wrapS = TextureWrapping::MirroredRepeat;
    texture->wrapT = TextureWrapping::Repeat;
    texture->anisotropy = 8;
    texture->repeat.set(2.f, 3.f);
    texture->offset.set(0.25f, 0.5f);
    texture->center.set(0.5f, 0.5f);
    texture->rotation = 0.75f;
    texture->colorSpace = ColorSpace::sRGB;

    auto material = MeshStandardMaterial::create();
    material->map = texture;

    auto mesh = Mesh::create(BoxGeometry::create(), material);
    mesh->name = "textured";

    ObjectExporter exporter;
    const auto text = exporter.toJson(*mesh);

    ObjectLoader loader;
    auto parsed = loader.parse(text);
    REQUIRE(parsed != nullptr);

    auto* parsedMaterial = parsed->materialAs<MeshStandardMaterial>();
    REQUIRE(parsedMaterial != nullptr);
    REQUIRE(parsedMaterial->map != nullptr);

    const auto& parsedTexture = *parsedMaterial->map;
    CHECK(parsedTexture.uuid() == texture->uuid());
    CHECK(parsedTexture.name == "swatch");
    CHECK(parsedTexture.wrapS == TextureWrapping::MirroredRepeat);
    CHECK(parsedTexture.wrapT == TextureWrapping::Repeat);
    CHECK(parsedTexture.magFilter == Filter::Nearest);
    CHECK(parsedTexture.minFilter == Filter::Nearest);
    CHECK(parsedTexture.anisotropy == 8);
    CHECK(parsedTexture.colorSpace == ColorSpace::sRGB);
    CHECK(parsedTexture.unpackAlignment == 1);
    CHECK_THAT(parsedTexture.repeat.x, WithinAbs(2.f, 1e-6));
    CHECK_THAT(parsedTexture.repeat.y, WithinAbs(3.f, 1e-6));
    CHECK_THAT(parsedTexture.offset.x, WithinAbs(0.25f, 1e-6));
    CHECK_THAT(parsedTexture.center.y, WithinAbs(0.5f, 1e-6));
    CHECK_THAT(parsedTexture.rotation, WithinAbs(0.75f, 1e-6));

    const auto& image = parsedTexture.image();
    CHECK(image.width() == 2);
    CHECK(image.height() == 2);
    CHECK(image.channels() == 4);
    CHECK(image.data<unsigned char>() == pixels);
}


TEST_CASE("three.js authored JSON parses with matching key names") {

    // Hand-authored in exactly the shape three.js r129 emits.
    const std::string threeJson = R"({
        "metadata": { "version": 4.5, "type": "Object", "generator": "Object3D.toJSON" },
        "geometries": [
            {
                "uuid": "GEOMETRY-1",
                "type": "BoxGeometry",
                "width": 2, "height": 3, "depth": 4,
                "widthSegments": 1, "heightSegments": 1, "depthSegments": 1
            }
        ],
        "materials": [
            {
                "uuid": "MATERIAL-1",
                "type": "MeshStandardMaterial",
                "color": 11579568,
                "roughness": 0.5,
                "metalness": 0.25,
                "emissive": 0,
                "envMapIntensity": 1,
                "depthFunc": 3, "depthTest": true, "depthWrite": true, "colorWrite": true,
                "stencilWrite": false, "stencilWriteMask": 255, "stencilFunc": 519,
                "stencilRef": 0, "stencilFuncMask": 255,
                "stencilFail": 7680, "stencilZFail": 7680, "stencilZPass": 7680
            }
        ],
        "object": {
            "uuid": "SCENE-1",
            "type": "Scene",
            "name": "Scene",
            "layers": 1,
            "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
            "background": 1118481,
            "children": [
                {
                    "uuid": "MESH-1",
                    "type": "Mesh",
                    "name": "Box",
                    "layers": 1,
                    "castShadow": true,
                    "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 1,2,3,1],
                    "geometry": "GEOMETRY-1",
                    "material": "MATERIAL-1"
                },
                {
                    "uuid": "LIGHT-1",
                    "type": "PointLight",
                    "name": "PointLight",
                    "layers": 1,
                    "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,5,0,1],
                    "color": 16777215,
                    "intensity": 0.8,
                    "distance": 12,
                    "decay": 2,
                    "shadow": {
                        "bias": -0.002,
                        "mapSize": [1024, 1024],
                        "camera": {
                            "uuid": "SHADOWCAM-1",
                            "type": "PerspectiveCamera",
                            "layers": 1,
                            "fov": 90, "zoom": 1, "near": 0.5, "far": 500,
                            "focus": 10, "aspect": 1, "filmGauge": 35, "filmOffset": 0
                        }
                    }
                },
                {
                    "uuid": "CAMERA-1",
                    "type": "PerspectiveCamera",
                    "name": "Camera",
                    "layers": 1,
                    "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,10,1],
                    "fov": 50, "zoom": 1, "near": 0.01, "far": 1000,
                    "focus": 10, "aspect": 1.5, "filmGauge": 35, "filmOffset": 0
                }
            ]
        }
    })";

    ObjectLoader loader;
    auto parsed = loader.parse(threeJson);
    REQUIRE(parsed != nullptr);

    auto* scene = parsed->as<Scene>();
    REQUIRE(scene != nullptr);
    CHECK(scene->uuid == "SCENE-1");
    REQUIRE(scene->background.isColor());
    CHECK(scene->background.color().getHex() == 0x111111);
    REQUIRE(scene->children.size() == 3);

    auto* mesh = findByUuid<Mesh>(*parsed, "MESH-1");
    REQUIRE(mesh != nullptr);
    CHECK(mesh->name == "Box");
    CHECK(mesh->castShadow);
    CHECK_THAT(mesh->position.x, WithinAbs(1.f, 1e-5));
    CHECK_THAT(mesh->position.y, WithinAbs(2.f, 1e-5));
    CHECK_THAT(mesh->position.z, WithinAbs(3.f, 1e-5));

    auto geometry = std::dynamic_pointer_cast<BoxGeometry>(mesh->geometry());
    REQUIRE(geometry != nullptr);
    CHECK_THAT(geometry->width, WithinAbs(2.f, 1e-5));
    CHECK_THAT(geometry->depth, WithinAbs(4.f, 1e-5));

    auto* material = mesh->materialAs<MeshStandardMaterial>();
    REQUIRE(material != nullptr);
    CHECK(material->color.getHex() == 0xb0b0b0);
    CHECK_THAT(material->roughness, WithinAbs(0.5f, 1e-6));
    CHECK_THAT(material->metalness, WithinAbs(0.25f, 1e-6));

    auto* light = findByUuid<PointLight>(*parsed, "LIGHT-1");
    REQUIRE(light != nullptr);
    CHECK_THAT(light->intensity, WithinAbs(0.8f, 1e-5));
    CHECK_THAT(light->distance, WithinAbs(12.f, 1e-5));
    CHECK_THAT(light->decay, WithinAbs(2.f, 1e-5));
    CHECK_THAT(light->shadow->bias, WithinAbs(-0.002f, 1e-7));
    CHECK_THAT(light->shadow->mapSize.x, WithinAbs(1024.f, 1e-6));

    auto* camera = findByUuid<PerspectiveCamera>(*parsed, "CAMERA-1");
    REQUIRE(camera != nullptr);
    CHECK_THAT(camera->fov, WithinAbs(50.f, 1e-5));
    CHECK_THAT(camera->aspect, WithinAbs(1.5f, 1e-5));
    CHECK_THAT(camera->nearPlane, WithinAbs(0.01f, 1e-6));
}


TEST_CASE("Parametric geometry survives export/import with its segment counts") {

    auto sphere = SphereGeometry::create(SphereGeometry::Params(2.5f, 24, 18, 0.25f, 3.f, 0.1f, 2.f));
    const auto originalTriangles = sphere->getIndex()->count() / 3;

    auto mesh = Mesh::create(sphere, MeshStandardMaterial::create());

    ObjectExporter exporter;
    ObjectLoader loader;
    auto parsed = loader.parse(exporter.toJson(*mesh));
    REQUIRE(parsed != nullptr);

    auto parsedGeometry = std::dynamic_pointer_cast<SphereGeometry>(parsed->geometry());
    REQUIRE(parsedGeometry != nullptr);
    CHECK(parsedGeometry->type() == "SphereGeometry");
    CHECK(parsedGeometry->uuid == sphere->uuid);
    CHECK_THAT(parsedGeometry->radius, WithinAbs(2.5f, 1e-5));
    CHECK(parsedGeometry->parameters.widthSegments == 24);
    CHECK(parsedGeometry->parameters.heightSegments == 18);
    CHECK_THAT(parsedGeometry->parameters.phiStart, WithinAbs(0.25f, 1e-5));
    CHECK_THAT(parsedGeometry->parameters.phiLength, WithinAbs(3.f, 1e-5));
    CHECK_THAT(parsedGeometry->parameters.thetaStart, WithinAbs(0.1f, 1e-5));
    CHECK_THAT(parsedGeometry->parameters.thetaLength, WithinAbs(2.f, 1e-5));

    REQUIRE(parsedGeometry->hasIndex());
    CHECK(parsedGeometry->getIndex()->count() / 3 == originalTriangles);
}


TEST_CASE("Malformed and unknown input is handled without crashing") {

    ObjectLoader loader;

    SECTION("garbage text returns nullptr") {

        CHECK(loader.parse("{ this is not json") == nullptr);
        CHECK(loader.parse("") == nullptr);
        CHECK(loader.parse("[1,2,3]") == nullptr);
    }

    SECTION("document without an object entry returns nullptr") {

        CHECK(loader.parse(R"({"metadata":{"version":4.5,"type":"Object"}})") == nullptr);
    }

    SECTION("unknown child type is skipped, the rest still loads") {

        const std::string text = R"({
            "metadata": { "version": 4.5, "type": "Object" },
            "object": {
                "uuid": "ROOT", "type": "Group", "layers": 1,
                "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                "children": [
                    { "uuid": "WAT", "type": "TotallyUnknownObject", "layers": 1,
                      "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] },
                    { "uuid": "OK", "type": "Group", "name": "kept", "layers": 1,
                      "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] }
                ]
            }
        })";

        auto parsed = loader.parse(text);
        REQUIRE(parsed != nullptr);
        REQUIRE(parsed->children.size() == 1);
        CHECK(parsed->children.front()->name == "kept");
    }

    SECTION("unknown geometry and material types degrade gracefully") {

        const std::string text = R"({
            "metadata": { "version": 4.5, "type": "Object" },
            "geometries": [ { "uuid": "G", "type": "MysteryGeometry", "radius": 1 } ],
            "materials": [ { "uuid": "M", "type": "MysteryMaterial", "color": 255 } ],
            "object": {
                "uuid": "ROOT", "type": "Mesh", "layers": 1,
                "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
                "geometry": "G", "material": "M"
            }
        })";

        auto parsed = loader.parse(text);
        REQUIRE(parsed != nullptr);
        // the unresolved geometry leaves the mesh with Mesh's empty default
        REQUIRE(parsed->geometry() != nullptr);
        CHECK(parsed->geometry()->getAttributes().empty());
        // three.js falls back to a default material rather than dropping the node
        REQUIRE(parsed->material() != nullptr);
        CHECK(parsed->material()->type() == "MeshStandardMaterial");
    }
}
