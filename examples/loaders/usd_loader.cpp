
#include "threepp/loaders/USDLoader.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>

using namespace threepp;

int main(int argc, char** argv) {
    Canvas canvas("USD loader example");
    auto renderer= createRenderer(canvas);

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

    std::string modelPath;
    if (argc > 1) {
        std::filesystem::path arg = argv[1];
        if (std::filesystem::exists(arg) && std::filesystem::is_regular_file(arg)) {
            auto ext = arg.extension().string();
            std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz") {
                modelPath = arg.string();
            } else {
                std::cerr << "Ignoring argument (not a .usd/.usda/.usdc/.usdz file): " << arg << "\n";
            }
        } else {
            std::cerr << "Ignoring argument (file not found): " << arg << "\n";
        }
    }

    if (!modelPath.empty()) {
        std::cout << "Loading: " << modelPath << "\n";
        auto model = loader.load(modelPath);
        if (!model) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        scene->add(model);
    } else {
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
    }

    auto floor = Mesh::create(
            BoxGeometry::create(10, 0.1f, 10),
            MeshStandardMaterial::create({{"color", Color::lightgray}}));
    floor->position.set(0, -2, 0);
    floor->receiveShadow = true;
    scene->add(floor);

    canvas.onWindowResize([&](WindowSize newSize) {
        renderer->setSize(newSize);
        camera->aspect = newSize.aspect();
        camera->updateProjectionMatrix();
    });

    canvas.animate([&] {
        renderer->render(*scene, *camera);
    });

}
