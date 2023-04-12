
#include "threepp/threepp.hpp"

using namespace threepp;

auto createBox(const Vector3& pos, const Color& color) {
    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshPhongMaterial::create({{"color", color}});
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.copy(pos);
    return box;
}

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto light = HemisphereLight::create();
    scene->add(light);

    auto group = Group::create();
    group->add(createBox({-1,0,0}, 0xff0000));
    group->add(createBox({1,0,0}, 0x00ff00));
    scene->add(group);

    auto group2 = group->clone(true);
    group2->position.z = -2;
    group2->rotateY(math::degToRad(180));
    group->add(group2);

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
