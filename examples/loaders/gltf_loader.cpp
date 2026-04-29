
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>

using namespace threepp;

int main(int argc, char** argv) {
    Canvas canvas("GLTF Demo");
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 2, -4);

    OrbitControls controls{*camera, canvas};

    RGBELoader hdrLoader;
    if (auto hdrTexture = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr" )) {
        scene->background = hdrTexture;
        scene->environment = hdrTexture;
    }

    auto ambientLight = AmbientLight::create(0xffffff, 0.2f);
    scene->add(ambientLight);

    auto dirLight = DirectionalLight::create(0xffffff, 1.0f);
    dirLight->position.set(1, 1, -1);
    dirLight->castShadow = true;
    scene->add(dirLight);

    std::string modelPath = std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb";
    if (argc > 1) {
        std::filesystem::path arg = argv[1];
        if (std::filesystem::exists(arg) && std::filesystem::is_regular_file(arg)) {
            auto ext = arg.extension().string();
            std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".gltf" || ext == ".glb") {
                modelPath = arg.string();
            } else {
                std::cerr << "Ignoring argument (not a .gltf/.glb file): " << arg << "\n";
            }
        } else {
            std::cerr << "Ignoring argument (file not found): " << arg << "\n";
        }
    }
    std::cout << "Loading: " << modelPath << "\n";

    GLTFLoader loader;
    auto result = loader.load(modelPath);
    result->scene->traverseType<Mesh>([&](Mesh& mesh) {
        mesh.castShadow = true;
    });

    if (!result) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    scene->add(result->scene);

    std::unique_ptr<AnimationMixer> mixer;
    if (!result->animations.empty()) {
        std::cout << "Loaded " << result->animations.size() << " animation clip(s)." << std::endl;
        mixer = std::make_unique<AnimationMixer>(*result->scene);
        mixer->clipAction(result->animations.front())->play();
    }

    auto skeletonHelper = SkeletonHelper::create(*result->scene);
    skeletonHelper->material()->as<LineBasicMaterial>()->linewidth = 2;
    scene->add(skeletonHelper);


    auto floor = Mesh::create(
            BoxGeometry::create(10, 0.1f, 10),
            MeshStandardMaterial::create({{"color", Color::lightgray}}));
    floor->position.set(0, 0, 0);
    floor->receiveShadow = true;
    scene->add(floor);

    canvas.onWindowResize([&](WindowSize newSize) {
        renderer->setSize(newSize);
        camera->aspect = newSize.aspect();
        camera->updateProjectionMatrix();
    });

    Clock clock;
    canvas.animate([&] {
        renderer->render(*scene, *camera);
        if (mixer) mixer->update(clock.getDelta());
    });

    return 0;
}
