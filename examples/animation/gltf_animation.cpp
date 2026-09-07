#include "renderer_factory.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>

using namespace threepp;

int main(int argc, char** argv) {

    // gltf_animation [model.gltf|.glb]
    // Pass a .gltf/.glb path to load your own model instead of the default
    // Soldier; the camera then frames it from its bounding box (mirrors gltf_loader).
    std::string modelPath = std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb";
    bool nonDefaultModelPath = false;
    for (int i = 1; i < argc; ++i) {
        std::filesystem::path arg = argv[i];
        if (std::filesystem::exists(arg) && std::filesystem::is_regular_file(arg)) {
            auto ext = arg.extension().string();
            std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".gltf" || ext == ".glb") {
                modelPath = arg.string();
                nonDefaultModelPath = true;
            } else {
                std::cerr << "Ignoring argument (not a .gltf/.glb file): " << arg << "\n";
            }
        } else {
            std::cerr << "Ignoring argument (file not found): " << arg << "\n";
        }
    }

    Canvas canvas("GLTF Animation", {{"aa", 4}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    PerspectiveCamera camera(45, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0, 2, -5);
    camera.lookAt({0, 1, 0});

    Scene scene;
    scene.background = Color(0x87ceeb);

    // Lights
    auto hemi = HemisphereLight::create(0xffffff, 0x8d8d8d, 1.f);
    hemi->position.set(0, 20, 0);
    scene.add(hemi);

    auto dir = DirectionalLight::create(0xffffff, 3.f);
    dir->position.set(3, 10, 10);
    dir->castShadow = true;
    scene.add(dir);

    // The default Soldier scene keeps its hand-tuned ground/grid/fog; a custom
    // model is shown bare and framed from its own bounding box instead (its
    // scale is unknown, so the fixed 20x20 ground/fog would rarely fit).
    if (!nonDefaultModelPath) {
        scene.fog = Fog(0x87ceeb, 20, 60);

        auto ground = Mesh::create(
                PlaneGeometry::create(20, 20),
                MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(0x999999).depthWrite(false)));
        ground->rotation.x = -math::PI / 2;
        ground->receiveShadow = true;
        scene.add(ground);

        auto grid = GridHelper::create(20, 20, 0x000000, 0x000000);
        grid->material()->opacity = 0.2f;
        grid->material()->transparent = true;
        scene.add(grid);
    }

    // Load GLTF model with animations via native GLTFLoader
    std::cout << "Loading: " << modelPath << "\n";
    GLTFLoader loader;
    auto result = loader.load(modelPath);

    if (!result) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    auto& model = result->scene;
    model->traverseType<Mesh>([](Mesh& m) {
        m.castShadow = true;
        m.receiveShadow = true;
    });
    scene.add(model);

    OrbitControls controls{camera, canvas};
    controls.target.set(0, 1, 0);

    if (nonDefaultModelPath) {
        // Scale the frustum and framing to the model's bounding box so near/far
        // clipping and camera distance stay sane whether it is centimeter- or
        // building-scale.
        Box3 bbox;
        bbox.setFromObject(*model);
        Vector3 modelCenter, modelSize;
        bbox.getCenter(modelCenter);
        bbox.getSize(modelSize);
        const float modelMaxDim = std::max({modelSize.x, modelSize.y, modelSize.z, 0.01f});

        camera.nearPlane = std::max(0.001f, modelMaxDim * 0.005f);
        camera.farPlane = modelMaxDim * 100.f;
        camera.updateProjectionMatrix();

        camera.position.set(modelCenter.x + 0.55f * modelMaxDim,
                            modelCenter.y + 0.30f * modelMaxDim,
                            modelCenter.z + 0.55f * modelMaxDim);
        camera.lookAt(modelCenter);
        controls.target.copy(modelCenter);
    }
    controls.update();

    auto& clips = result->animations;

    std::cout << "Loaded " << clips.size() << " animation(s):\n";
    for (auto& clip : clips) std::cout << "  - " << clip->name() << "\n";

    auto skeletonHelper = SkeletonHelper::create(*model);
    skeletonHelper->visible = false;
    scene.add(skeletonHelper);

    // Animation mixer — root is the scene returned by GLTF loader. A model with
    // no clips (e.g. a custom static mesh) still displays; there is just nothing
    // to play or crossfade.
    std::unique_ptr<AnimationMixer> mixer;
    AnimationAction* currentAction = nullptr;
    int currentClip = 0;
    if (!clips.empty()) {
        mixer = std::make_unique<AnimationMixer>(*model);
        currentAction = mixer->clipAction(clips[currentClip]);
        currentAction->play();
    } else {
        std::cout << "(model has no animation clips)\n";
    }

    // Crossfade to next animation
    float fadeDuration = 0.5f;
    float clipTimer = 0.f;
    float clipHoldTime = 3.f;

    auto crossfadeToNext = [&] {
        if (!mixer || clips.size() < 2) return;
        auto* prevAction = currentAction;
        currentClip = (currentClip + 1) % static_cast<int>(clips.size());
        currentAction = mixer->clipAction(clips[currentClip]);
        currentAction->reset();
        currentAction->play();
        prevAction->crossFadeTo(currentAction, fadeDuration);
        std::cout << "Switching to: " << clips[currentClip]->name() << "\n";
    };

    // Keyboard: S = toggle skeleton, Space = next animation
    canvas.onKeyPressed([&](KeyEvent evt) {
        if (evt.key == Key::S) {
            skeletonHelper->visible = !skeletonHelper->visible;
        } else if (evt.key == Key::SPACE) {
            clipTimer = 0.f;
            crossfadeToNext();
        }
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer->setSize(size);
    });

    std::cout << "Press SPACE to switch animation, S to toggle skeleton\n";

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        clipTimer += dt;
        if (clipTimer >= clipHoldTime && clips.size() > 1) {
            clipTimer = 0.f;
            crossfadeToNext();
        }

        if (mixer) mixer->update(dt);
        renderer->render(scene, camera);
    });
}
