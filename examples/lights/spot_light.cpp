
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/helpers/SpotLightHelper.hpp"
#include "threepp/threepp.hpp"

#include <cmath>

using namespace threepp;

namespace {

    auto createTorusKnot() {
        const auto geometry = TorusKnotGeometry::create(0.75f, 0.2f, 128, 64);
        const auto material = MeshStandardMaterial::create();
        material->roughness = 0.1;
        material->metalness = 0.1;
        material->color = 0xff0000;
        material->emissive = 0x000000;
        auto knot = Mesh::create(geometry, material);
        knot->castShadow = true;
        knot->position.y = 4;
        knot->scale *= 2;

        return knot;
    }

    auto createPlane() {
        const auto planeGeometry = PlaneGeometry::create(150, 150);
        const auto planeMaterial = MeshPhongMaterial::create({{"color", Color::gray}, {"side", Side::Double}});
        auto plane = Mesh::create(planeGeometry, planeMaterial);
        plane->position.setY(-1);
        plane->rotateX(math::degToRad(-90));
        plane->receiveShadow = true;

        return plane;
    }

}// namespace

int main() {

    Canvas canvas("SpotLight", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.set(0, 10, 25);

    OrbitControls controls{*camera, canvas};

    auto light = SpotLight::create(Color::peachpuff);
    light->distance = 30;
    light->angle = math::degToRad(20);
    light->position.set(10, 10, 0);
    light->castShadow = true;
    scene->add(light);

    scene->add(AmbientLight::create(0xffffff, 0.1f));

    auto helper = SpotLightHelper::create(*light);
    scene->add(helper);

    auto target = Object3D::create();
    light->setTarget(*target);
    scene->add(target);

    auto knot = createTorusKnot();
    scene->add(knot);

    auto plane = createPlane();
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();
        const auto t = clock.elapsedTime;

        knot->rotation.y += 0.5f * dt;

        target->position.x = 5 * std::sin(t);
        target->position.z = 5 * std::cos(t);

        helper->update();

        renderer.render(*scene, *camera);
    });
}
