
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

    auto addLights(Scene& scene) {
        const auto light1 = PointLight::create(Color::yellow);
        light1->castShadow = true;
        light1->shadow->bias = -0.005f;
        light1->distance = 8;
        light1->position.y = 4;

        const auto light2 = PointLight::create(Color::white);
        light2->castShadow = true;
        light2->shadow->bias = -0.005f;
        light2->distance = 8;
        light2->position.y = 4;

        const auto light3 = PointLight::create(Color::purple);
        light3->castShadow = true;
        light3->shadow->bias = -0.005f;
        light3->distance = 10;
        light3->position.y = 7;

        const auto lightHelper1 = PointLightHelper::create(*light1, 0.25f);
        const auto lightHelper2 = PointLightHelper::create(*light2, 0.25f);
        const auto lightHelper3 = PointLightHelper::create(*light3, 0.25f);

        light1->name = "light1";
        light2->name = "light2";
        light3->name = "light3";

        scene.add(light1);
        scene.add(light2);
        scene.add(light3);

        scene.add(lightHelper1);
        scene.add(lightHelper2);
        scene.add(lightHelper3);
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

    addLights(*scene);

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

    auto light1 = scene->getObjectByName("light1");
    auto light2 = scene->getObjectByName("light2");

    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();
        const auto t = clock.elapsedTime;

        knot->rotation.y += 0.5f * dt;

        light1->position.x = 2 * std::sin(t);
        light1->position.z = 7 * std::cos(t);

        light2->position.x = 5 * std::sin(t);
        light2->position.z = 1 * std::sin(t);

        renderer.render(*scene, *camera);
    });
}
