// Object3D::clone() builds the copy through createDefault(), which returns a
// plain Object3D unless the class overrides it. Classes that did not came back
// as their nearest overriding base (a LineSegments as a Line, drawn as a
// strip) or as a bare Object3D (lights, cameras, scenes), and the editor's
// Duplicate command goes through clone().

#include <catch2/catch_test_macros.hpp>

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/lights/SpotLightShadow.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LineLoop.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Scene.hpp"

#include <typeinfo>
#include <vector>

using namespace threepp;

TEST_CASE("clone keeps the concrete type") {

    const auto geometry = BoxGeometry::create();

    const std::vector<std::shared_ptr<Object3D>> objects{
            Group::create(),
            Mesh::create(geometry, MeshBasicMaterial::create()),
            Line::create(geometry, LineBasicMaterial::create()),
            LineSegments::create(geometry, LineBasicMaterial::create()),
            LineLoop::create(geometry, LineBasicMaterial::create()),
            Points::create(geometry),
            Sprite::create(),
            InstancedMesh::create(geometry, MeshBasicMaterial::create(), 3),
            SkinnedMesh::create(geometry, MeshBasicMaterial::create()),
            PerspectiveCamera::create(),
            OrthographicCamera::create(),
            AmbientLight::create(),
            HemisphereLight::create(),
            DirectionalLight::create(),
            PointLight::create(),
            SpotLight::create(),
            RectAreaLight::create(),
            Scene::create(),
    };

    for (const auto& object : objects) {

        INFO(object->type());

        const auto clone = object->clone();
        REQUIRE(clone != nullptr);
        CHECK(typeid(*clone) == typeid(*object));
        CHECK(clone->type() == object->type());
    }
}

TEST_CASE("cameras clone their projection") {

    auto perspective = PerspectiveCamera::create(35, 1.5f, 0.3f, 400);
    perspective->zoom = 2;
    perspective->focus = 4;
    perspective->filmGauge = 24;
    perspective->setViewOffset(200, 100, 10, 20, 50, 40);

    const auto p = perspective->clone<PerspectiveCamera>();
    REQUIRE(p != nullptr);
    CHECK(p->fov == 35);
    CHECK(p->aspect == 2);// setViewOffset sets it to fullWidth / fullHeight
    CHECK(p->nearPlane == 0.3f);
    CHECK(p->farPlane == 400);
    CHECK(p->zoom == 2);
    CHECK(p->focus == 4);
    CHECK(p->filmGauge == 24);
    REQUIRE(p->view.has_value());
    CHECK(p->view->offsetX == 10);
    CHECK(p->projectionMatrix == perspective->projectionMatrix);

    auto ortho = OrthographicCamera::create(-3, 3, 2, -2, 1, 50);
    const auto o = ortho->clone<OrthographicCamera>();
    REQUIRE(o != nullptr);
    CHECK(o->left == -3);
    CHECK(o->right == 3);
    CHECK(o->top == 2);
    CHECK(o->bottom == -2);
    CHECK(o->farPlane == 50);
    CHECK(o->projectionMatrix == ortho->projectionMatrix);
}

TEST_CASE("lights clone their shadow settings and target") {

    Object3D aim;

    auto directional = DirectionalLight::create(0xff0000, 3);
    directional->castShadow = true;
    directional->shadow->bias = -0.001f;
    directional->shadow->normalBias = 0.02f;
    directional->shadow->radius = 4;
    directional->shadow->mapSize.set(512, 256);
    auto* shadowCamera = directional->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = -40;
    shadowCamera->right = 40;
    shadowCamera->farPlane = 90;
    directional->setTarget(aim);

    const auto d = directional->clone<DirectionalLight>();
    REQUIRE(d != nullptr);
    CHECK(d->intensity == 3);
    CHECK(d->castShadow);
    CHECK(d->shadow != directional->shadow);
    CHECK(d->shadow->bias == -0.001f);
    CHECK(d->shadow->normalBias == 0.02f);
    CHECK(d->shadow->radius == 4);
    CHECK(d->shadow->mapSize == Vector2(512, 256));
    const auto* dCamera = d->shadow->camera->as<OrthographicCamera>();
    REQUIRE(dCamera != nullptr);
    CHECK(dCamera->left == -40);
    CHECK(dCamera->right == 40);
    CHECK(dCamera->farPlane == 90);
    CHECK(&d->target() == &aim);

    auto spot = SpotLight::create(0xffffff, 1, 12, 0.4f, 0.2f, 2);
    dynamic_cast<SpotLightShadow&>(*spot->shadow).focus = 0.5f;
    spot->setTarget(aim);

    const auto s = spot->clone<SpotLight>();
    REQUIRE(s != nullptr);
    CHECK(s->distance == 12);
    CHECK(s->angle == 0.4f);
    CHECK(dynamic_cast<SpotLightShadow&>(*s->shadow).focus == 0.5f);
    CHECK(&s->target() == &aim);

    auto point = PointLight::create(0x00ff00, 5, 20, 2);
    point->shadow->mapSize.set(128, 128);

    const auto pl = point->clone<PointLight>();
    REQUIRE(pl != nullptr);
    CHECK(pl->color == Color(0x00ff00));
    CHECK(pl->intensity == 5);
    CHECK(pl->distance == 20);
    CHECK(pl->decay == 2);
    CHECK(pl->shadow->mapSize == Vector2(128, 128));
}

TEST_CASE("a RectAreaLight clone keeps its size and has one quad") {

    auto light = RectAreaLight::create(0xffffff, 2, 3, 0.5f);
    auto marker = Group::create();
    marker->name = "marker";
    light->add(marker);

    const auto clone = light->clone<RectAreaLight>();
    REQUIRE(clone != nullptr);
    CHECK(clone->width == 3);
    CHECK(clone->height == 0.5f);
    CHECK(clone->intensity == 2);

    // Its own quad plus the cloned marker, not a second quad.
    REQUIRE(clone->children.size() == light->children.size());
    CHECK(clone->getObjectByName("marker") != nullptr);
    CHECK(clone->getObjectByName("marker") != marker.get());
}

TEST_CASE("an InstancedMesh clone copies its instances") {

    auto mesh = InstancedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create(), 4);
    Matrix4 m;
    m.makeTranslation(1, 2, 3);
    mesh->setMatrixAt(2, m);
    mesh->setColorAt(1, Color(0xff8000));
    mesh->setCount(3);

    const auto clone = mesh->clone<InstancedMesh>();
    REQUIRE(clone != nullptr);
    CHECK(clone->geometry() == mesh->geometry());
    CHECK(clone->count() == 3);

    Matrix4 read;
    clone->getMatrixAt(2, read);
    CHECK(read == m);

    Color color;
    clone->getColorAt(1, color);
    CHECK(color == Color(0xff8000));

    // The instance buffers are the clone's own.
    CHECK(clone->instanceMatrix() != mesh->instanceMatrix());
    CHECK(clone->instanceColor() != mesh->instanceColor());
    mesh->setMatrixAt(2, Matrix4());
    clone->getMatrixAt(2, read);
    CHECK(read == m);
}

TEST_CASE("a SkinnedMesh clone shares the skeleton and keeps the bind matrices") {

    auto bone = Bone::create();
    auto skeleton = Skeleton::create({bone});

    auto mesh = SkinnedMesh::create(BoxGeometry::create(), MeshBasicMaterial::create());
    Matrix4 bindMatrix;
    bindMatrix.makeTranslation(0, 1, 0);
    mesh->bind(skeleton, bindMatrix);
    mesh->bindMode = SkinnedMesh::BindMode::Detached;

    const auto clone = mesh->clone<SkinnedMesh>();
    REQUIRE(clone != nullptr);
    CHECK(clone->skeleton == skeleton);
    CHECK(clone->bindMode == SkinnedMesh::BindMode::Detached);
    CHECK(clone->bindMatrix == mesh->bindMatrix);
    CHECK(clone->bindMatrixInverse == mesh->bindMatrixInverse);
}

TEST_CASE("a Scene clone keeps background and fog") {

    auto scene = Scene::create();
    scene->background = Color(0x102030);
    scene->fog = Fog(0x405060, 2, 80);
    scene->autoUpdate = false;
    scene->add(Mesh::create(BoxGeometry::create(), MeshBasicMaterial::create()));

    const auto clone = scene->clone<Scene>();
    REQUIRE(clone != nullptr);
    REQUIRE(clone->background.isColor());
    CHECK(clone->background.color() == Color(0x102030));
    REQUIRE(clone->fog.has_value());
    CHECK(std::get<Fog>(*clone->fog) == Fog(0x405060, 2, 80));
    CHECK_FALSE(clone->autoUpdate);
    CHECK(clone->children.size() == 1);
}
