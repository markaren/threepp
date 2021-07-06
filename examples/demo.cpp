
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
//    renderer.setClearColor(Color(0xffffff));
    renderer.setSize(canvas.getWidth(), canvas.getHeight());
//    renderer.sortObjects = true;

//    const auto light = AmbientLight::create(0xffffff);
//    scene->add(light);

    const auto geometry = SphereGeometry::create(2);
    const auto material = MeshBasicMaterial::create(/*{ color: 0x00ff00 }*/);
    material->color.setHex(0xff0000);
    auto cube = Mesh::create(geometry, material);
    scene->add(cube);

    camera->position.z = 5;

    canvas.animate([&](float dt) {
//        cube->rotation.x(cube->rotation.x() + 0.01f);
//        cube->rotation.y(cube->rotation.y() + 0.01f);
camera->rotation.y(camera->rotation.y() + 0.01f);

        renderer.render(scene, camera);
    });
}
