
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/helpers/PointLightHelper.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    auto createTorusKnot() {
        const auto geometry = TorusKnotGeometry::create(0.75f, 0.2f, 128, 64);
        const auto material = MeshStandardMaterial::create();
        material->roughness = 0.1f;
        material->metalness = 0.1f;
        material->color = 0xff0000;
        material->emissive = 0x000000;
        auto knot = Mesh::create(geometry, material);
        knot->castShadow = true;
        knot->position.y = 1;

        return knot;
    }

    auto createPlane() {
        const auto planeGeometry = PlaneGeometry::create(105, 105);
        const auto planeMaterial = MeshPhongMaterial::create();
        planeMaterial->color.setHex(Color::white);
        planeMaterial->side = Side::Double;
        auto plane = Mesh::create(planeGeometry, planeMaterial);
        plane->receiveShadow = true;
        plane->rotateX(math::degToRad(-90));

        return plane;
    }

}// namespace

int main() {

    Canvas canvas("PointLight", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.set(5, 4, 6);

    OrbitControls controls{*camera, canvas};

    auto light1 = PointLight::create(Color::yellow);
    light1->castShadow = true;
    light1->shadow->bias = -0.005f;
    light1->distance = 8;
    light1->position.y = 4;
    scene->add(light1);

    auto lightHelper1 = PointLightHelper::create(*light1, 0.25f);
    scene->add(lightHelper1);

    auto light2 = PointLight::create(Color::white);
    light2->castShadow = true;
    light2->shadow->bias = -0.005f;
    light2->distance = 8;
    light2->position.y = 4;
    scene->add(light2);

    auto lightHelper2 = PointLightHelper::create(*light2, 0.25f);
    scene->add(lightHelper2);

    auto knot = createTorusKnot();
    scene->add(knot);

    auto plane = createPlane();
    plane->position.y = -1;
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();
        float t = clock.elapsedTime;

        knot->rotation.y += 0.5f * dt;

        light1->position.x = 2 * std::sin(t);
        light1->position.z = 7 * std::cos(t);

        light2->position.x = 5 * std::sin(t);
        light2->position.z = 1 * std::sin(t);

        renderer.render(*scene, *camera);
    });
}
