
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);
    renderer.shadowMap.enabled = true;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 3, 5);

    OrbitControls controls{camera, canvas};

    auto light1 = PointLight::create(Color::yellow);
    light1->castShadow = true;
    light1->shadow->bias = -0.005;
    light1->distance = 5;
    light1->position.y = 2;
    scene->add(light1);

    auto lightHelper1 = PointLightHelper::create(light1, 0.25f);
    scene->add(lightHelper1);

    auto light2 = PointLight::create(Color::white);
    light2->castShadow = true;
    light2->shadow->bias = -0.005;
    light2->distance = 5;
    light2->position.y = 2;
    scene->add(light2);

    auto lightHelper2 = PointLightHelper::create(light2, 0.25f);
    scene->add(lightHelper2);

    auto group = Group::create();
    scene->add(group);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshLambertMaterial::create();
    boxMaterial->color.setHex(0xff0000);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->castShadow = true;
    box->position.x = -1;
    group->add(box);

    auto box2 = Mesh::create(boxGeometry, boxMaterial->clone());
    box2->material()->as<MaterialWithColor>()->color.setHex(0x00ff00);
    box2->castShadow = true;
    box2->position.x = 1;
    group->add(box2);


    const auto planeGeometry = PlaneGeometry::create(105, 105);
    const auto planeMaterial = MeshPhongMaterial::create();
    planeMaterial->color.setHex(Color::white);
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.y = -1;
    plane->receiveShadow = true;
    plane->rotateX(math::degToRad(-90));
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float t, float dt) {
        group->rotation.y += 0.5f * dt;

        light1->position.x = 2 * std::sin(t);
        light1->position.z = 7 * std::cos(t);

        light2->position.x = 5 * std::sin(t);
        light2->position.z = 1 * std::sin(t);

        renderer.render(scene, camera);
    });
}
