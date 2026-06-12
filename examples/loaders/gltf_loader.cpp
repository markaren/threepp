
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
    renderer->toneMapping = ToneMapping::ACESFilmic;
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

    auto dirLight = DirectionalLight::create(0xffffff, 1.0f);
    dirLight->position.set(1, 1, -1);
    dirLight->castShadow = true;
    scene->add(dirLight);

    // gltf_loader [model.gltf|.glb] [--shot out.png] [--frames N] [--zoom f]
    // --shot: headless capture — frame the loaded model from its bounding box,
    // render N warm-up frames (TAA/denoiser converge), write one PNG, exit.
    std::string modelPath = std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb";
    std::string shotPath;
    int   shotFrames = 240;
    float shotZoom   = 1.f; // <1 = closer (distance scales with it)
    bool  shotFront  = false;// view head-on along the model's thinnest axis (test walls/panels)
    std::string shotAxis;    // explicit --front axis: x, -x, z, -z
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc) {
            shotPath = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            shotFrames = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--zoom") == 0 && i + 1 < argc) {
            shotZoom = static_cast<float>(std::atof(argv[++i]));
            continue;
        }
        if (std::strcmp(argv[i], "--front") == 0) {
            shotFront = true;
            // optional axis token: x, -x, z, -z (default: thinnest bbox axis)
            if (i + 1 < argc && (argv[i + 1][0] == 'x' || argv[i + 1][0] == 'z' || argv[i + 1][0] == '-')) {
                shotAxis = argv[++i];
            }
            continue;
        }
        std::filesystem::path arg = argv[i];
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
    const bool capturing = !shotPath.empty();
    int shotFrame = 0;
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
    skeletonHelper->materialAs<LineBasicMaterial>()->linewidth = 2;
    scene->add(skeletonHelper);


    auto floor = Mesh::create(
            BoxGeometry::create(10, 0.1f, 10),
            MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::lightgray)));
    floor->position.set(0, 0, 0);
    floor->receiveShadow = true;
    scene->add(floor);

    if (capturing) {
        // Frame the MODEL (not the whole scene — the helper floor would widen
        // the bounds) from a low oblique view.
        Box3 bbox;
        bbox.setFromObject(*result->scene);
        Vector3 center, size;
        bbox.getCenter(center);
        bbox.getSize(size);
        const float maxDim = std::max({size.x, size.y, size.z, 0.01f});
        const float d = maxDim * shotZoom;
        if (shotFront) {
            // Head-on along the thinnest bbox axis (a wall/panel's facing),
            // or the explicitly requested axis.
            Vector3 dir = (size.z <= size.x) ? Vector3(0, 0, 1) : Vector3(1, 0, 0);
            if (shotAxis == "x") dir.set(1, 0, 0);
            else if (shotAxis == "-x") dir.set(-1, 0, 0);
            else if (shotAxis == "z") dir.set(0, 0, 1);
            else if (shotAxis == "-z") dir.set(0, 0, -1);
            camera->position.copy(center).addScaledVector(dir, 0.9f * d);
            camera->lookAt(center);
            std::cout << "[shot] axis=" << (shotAxis.empty() ? "auto" : shotAxis)
                      << " center=(" << center.x << "," << center.y << "," << center.z
                      << ") size=(" << size.x << "," << size.y << "," << size.z
                      << ") cam=(" << camera->position.x << "," << camera->position.y
                      << "," << camera->position.z << ")\n";
        } else {
            camera->position.set(center.x + 0.55f * d,
                                 center.y + 0.30f * d,
                                 center.z + 0.55f * d);
            camera->lookAt(Vector3(center.x, center.y + 0.05f * d, center.z));
        }
        camera->updateMatrixWorld();
    }

    canvas.onWindowResize([&](WindowSize newSize) {
        renderer->setSize(newSize);
        camera->aspect = newSize.aspect();
        camera->updateProjectionMatrix();
    });

    Clock clock;
    canvas.animate([&] {
        renderer->render(*scene, *camera);
        if (mixer) mixer->update(clock.getDelta());
        if (capturing && ++shotFrame >= shotFrames) {
            renderer->writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << "\n";
            std::exit(0);
        }
    });

    return 0;
}
