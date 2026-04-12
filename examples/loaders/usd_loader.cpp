
#include "threepp/loaders/USDLoader.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main() {
    Canvas canvas("USD loader example");
    GLRenderer renderer{canvas.size()};

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000.f);
    camera->position.set(0, 1, 5);

    OrbitControls controls{*camera, canvas};

    auto ambientLight = AmbientLight::create(0xffffff, 0.2f);
    scene->add(ambientLight);

    auto dirLight = DirectionalLight::create(0xffffff, 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    USDLoader loader;
    auto catPlane = loader.load(std::string(DATA_FOLDER) + "/models/usd/texture-cat-plane.usdz");

    if (!catPlane) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    catPlane->position.x = -2;
    scene->add(catPlane);

    auto suzanne = loader.load(std::string(DATA_FOLDER) + "/models/usd/suzanne-pbr.usda");

    if (!suzanne) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    suzanne->position.x = 2;
    scene->add(suzanne);

    auto floor = Mesh::create(
            BoxGeometry::create(10, 0.1f, 10),
            MeshStandardMaterial::create({{"color", Color::lightgray}}));
    floor->position.set(0, -2, 0);
    floor->receiveShadow = true;
    scene->add(floor);

    canvas.onWindowResize([&](WindowSize newSize) {
        renderer.setSize(newSize);
        camera->aspect = newSize.aspect();
        camera->updateProjectionMatrix();
    });

    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
