
#include "threepp/loaders/FBXLoader.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main(int argc, char* argv[]) {
    const std::string defaultPath = std::string(DATA_FOLDER) + "/models/fbx/stanford-bunny.fbx";
    const std::string fbxPath = argc > 1 ? argv[1] : defaultPath;

    Canvas canvas("FBX loader example");
    GLRenderer renderer{canvas.size()};

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000000.f);
    camera->position.set(0, 2, 10);

    OrbitControls controls{*camera, canvas};

    auto ambientLight = AmbientLight::create(0xffffff, 0.2f);
    scene->add(ambientLight);

    auto dirLight = DirectionalLight::create(0xffffff, 1.0f);
    dirLight->position.set(1, 1, 1);
    scene->add(dirLight);

    FBXLoader loader;
    auto model = loader.load(fbxPath);

    if (!model) {
        std::cerr << "Failed to load FBX: " << fbxPath << "\n";
        return 1;
    }
    scene->add(model);

    Box3 bb;
    bb.setFromObject(*model);

    camera->position.set(0, bb.getSize().y * 2, bb.getSize().z * 2);
    camera->lookAt(bb.getCenter());

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
