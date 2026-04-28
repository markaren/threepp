
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    auto createKnot() {
        const auto geometry = TorusKnotGeometry::create(1.0f, 0.3f, 128, 32);
        const auto material = MeshStandardMaterial::create();
        material->color = 0xffffff;
        material->roughness = 0.1f;
        material->metalness = 0.9f;
        auto knot = Mesh::create(geometry, material);
        knot->position.y = 2.5f;
        return knot;
    }

    auto createBox() {
        const auto geometry = BoxGeometry::create(2, 2, 2);
        const auto material = MeshStandardMaterial::create();
        material->color = 0xffffff;
        material->roughness = 0.3f;
        material->metalness = 0.0f;
        auto box = Mesh::create(geometry, material);
        box->position.set(4, 1, 0);
        return box;
    }

    auto createFloor() {
        const auto geometry = PlaneGeometry::create(40, 40);
        const auto material = MeshStandardMaterial::create();
        material->color = 0x808080;
        material->roughness = 0.2f;
        material->metalness = 0.1f;
        auto plane = Mesh::create(geometry, material);
        plane->rotateX(math::degToRad(-90));
        return plane;
    }

    std::shared_ptr<RectAreaLight> createLight(const Color& color, const Vector3& pos, const Vector3& lookAt) {
        auto light = RectAreaLight::create(color, 5.f, 4.f, 4.f);
        light->position.copy(pos);
        light->lookAt(lookAt);
        return light;
    }

}// namespace

int main() {

    Canvas canvas("RectAreaLight", {{"aa", 4}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color(0x101016);

    auto camera = PerspectiveCamera::create(45, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 6, -16);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 2, 0);
    controls.update();

    scene->add(AmbientLight::create(0xffffff, 0.02f));

    auto light1 = createLight(0xff0000, {-5, 5, 5}, {0, 0, 0});
    scene->add(light1);

    auto light2 = createLight(0x00ff00, {0, 5, 5}, {0, 0, 0});
    scene->add(light2);

    auto light3 = createLight(0x0000ff, {5, 5, 5}, {0, 0, 0});
    scene->add(light3);

    auto knot = createKnot();
    scene->add(knot);

    auto box = createBox();
    scene->add(box);

    auto floor = createFloor();
    scene->add(floor);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const auto dt = clock.getDelta();
        knot->rotation.y += 0.3f * dt;
        knot->rotation.x += 0.2f * dt;

        renderer->render(*scene, *camera);
    });
}
