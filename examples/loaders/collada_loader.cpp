
#include "threepp/loaders/ColladaLoader.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main() {
    Canvas canvas("Collada Demo");
    GLRenderer renderer{canvas.size()};
    renderer.shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 5, 5);

    OrbitControls controls{*camera, canvas};

    auto ambientLight = AmbientLight::create(0xffffff, 0.2f);
    scene->add(ambientLight);


    auto dirLight = DirectionalLight::create(0xffffff, 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    ColladaLoader loader;
    auto stormTrooper = loader.load(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");

    if (!stormTrooper) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    scene->add(stormTrooper);

    Box3 bb;
    bb.setFromObject(*stormTrooper);


    controls.target = bb.getCenter();
    controls.update();

    canvas.animate([&] {
        renderer.render(*scene, *camera);
    });

    return 0;
}
