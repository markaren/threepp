
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.shadowMap.enabled = true;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(0, 1, 5);

    OrbitControls controls{camera, canvas};

    auto light = PointLight::create();
    light->castShadow = true;
    light->shadow->bias = -0.005;
    light->distance = 10;
    light->position.set(0, 1, 0);
    scene->add(light);

    scene->add(AmbientLight::create(0xffffff, 0.1f));

    auto helper = PointLightHelper::create(light, 0.25f);
    scene->add(helper);

    auto group = Group::create();
    scene->add(group);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshPhongMaterial::create();
    boxMaterial->color.setHex(0xff0000);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->castShadow = true;
    box->position.setX(-1);
    group->add(box);

    auto box2 = Mesh::create(boxGeometry, boxMaterial->clone());
    box2->material()->as<MaterialWithColor>()->color.setHex(0x00ff00);
    box2->castShadow = true;
    box2->position.setX(1);
    group->add(box2);


    const auto planeGeometry = PlaneGeometry::create(105, 105);
    const auto planeMaterial = MeshPhongMaterial::create();
    planeMaterial->color.setHex(Color::white);
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setY(-1);
    plane->receiveShadow = true;
    plane->rotateX(math::degToRad(-90));
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
