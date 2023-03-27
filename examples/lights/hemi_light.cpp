
#include "threepp/threepp.hpp"
#include "threepp/helpers/HemisphereLightHelper.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto light = HemisphereLight::create(0xffffbb, 0x082820);
    light->position.y = 2;
    scene->add(light);

    auto helper = HemisphereLightHelper::create(*light, 0.5f);
    scene->add(helper);

    auto group = Group::create();

    {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create({{"color", 0xff0000}});
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.x = -1;
        group->add(box);
    }

    {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create({{"color", 0x00ff00}});
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.x = 1;
        group->add(box);
    }

    scene->add(group);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshPhongMaterial::create({{"color", Color::gray}, {"side", DoubleSide}});
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.y = -1;
    plane->rotateX(math::degToRad(90));
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
      group->rotation.y += 0.5f * dt;

      renderer.render(scene, camera);
    });

}
