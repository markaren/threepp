
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.shadowMap.enabled = true;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto light = SpotLight::create(Color::peachpuff);
    light->distance = 30;
    light->angle = math::degToRad(20);
    light->position.set(10, 10, 0);
    light->castShadow = true;
    scene->add(light);

    scene->add(AmbientLight::create(0xffffff, 0.1f));

    auto helper = SpotLightHelper::create(light);
    scene->add(helper);

    auto target = Object3D::create();
    light->target = target;
    scene->add(target);

    auto group = Group::create();

    {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create();
        boxMaterial->color.setHex(0xff0000);
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.setX(-1);
        box->castShadow = true;
        group->add(box);
    }

    {
        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshPhongMaterial::create();
        boxMaterial->color.setHex(0x00ff00);
        auto box = Mesh::create(boxGeometry, boxMaterial);
        box->position.setX(1);
        box->castShadow = true;
        group->add(box);
    }

    scene->add(group);

    const auto planeGeometry = PlaneGeometry::create(150, 150);
    const auto planeMaterial = MeshPhongMaterial::create();
    planeMaterial->color.setHex(Color::gray);
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setY(-1);
    plane->rotateX(math::degToRad(-90));
    plane->receiveShadow = true;
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
      group->rotation.y += 0.5f * dt;

      target->position.x = 5 * std::sin(t);
      target->position.z = 5 * std::cos(t);

      helper->update();

      renderer.render(scene, camera);
    });

}
