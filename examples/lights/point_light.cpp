
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/helpers/PointLightHelper.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(4));
    GLRenderer renderer(canvas);
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 3, 5);

    OrbitControls controls{camera, canvas};

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

    const auto geometry = TorusKnotGeometry::create(0.75f, 0.2f, 128, 64);
    const auto material = MeshStandardMaterial::create();
    material->roughness = 0.1f;
    material->metalness = 0.1f;
    material->color = 0xff0000;
    material->emissive = 0x000000;
    auto mesh = Mesh::create(geometry, material);
    mesh->castShadow = true;
    mesh->position.y = 1;
    scene->add(mesh);

    const auto planeGeometry = PlaneGeometry::create(105, 105);
    const auto planeMaterial = MeshPhongMaterial::create();
    planeMaterial->color.setHex(Color::white);
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->receiveShadow = true;
    plane->rotateX(math::degToRad(-90));
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
        mesh->rotation.y += 0.5f * dt;

        light1->position.x = 2 * std::sin(t);
        light1->position.z = 7 * std::cos(t);

        light2->position.x = 5 * std::sin(t);
        light2->position.z = 1 * std::sin(t);

        renderer.render(scene, camera);
    });
}
