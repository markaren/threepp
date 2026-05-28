#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main() {

    Canvas canvas("GLTF Animation", {{"aa", 4}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    PerspectiveCamera camera(45, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0, 2, -5);
    camera.lookAt({0, 1, 0});

    Scene scene;
    scene.background = Color(0x87ceeb);
    scene.fog = Fog(0x87ceeb, 20, 60);

    // Ground
    auto ground = Mesh::create(
            PlaneGeometry::create(20, 20),
            MeshPhongMaterial::create({{"color", 0x999999}, {"depthWrite", false}}));
    ground->rotation.x = -math::PI / 2;
    ground->receiveShadow = true;
    scene.add(ground);

    auto grid = GridHelper::create(20, 20, 0x000000, 0x000000);
    grid->material()->opacity = 0.2f;
    grid->material()->transparent = true;
    scene.add(grid);

    // Lights
    auto hemi = HemisphereLight::create(0xffffff, 0x8d8d8d, 1.f);
    hemi->position.set(0, 20, 0);
    scene.add(hemi);

    auto dir = DirectionalLight::create(0xffffff, 3.f);
    dir->position.set(3, 10, 10);
    dir->castShadow = true;
    scene.add(dir);

    // Load GLTF model with animations via native GLTFLoader
    GLTFLoader loader;
    auto result = loader.load(std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb");

    if (!result || result->animations.empty()) {
        std::cerr << "Failed to load model or no animations found\n";
        return 1;
    }

    auto& model = result->scene;
    model->traverseType<Mesh>([](Mesh& m) {
        m.castShadow = true;
        m.receiveShadow = true;
    });
    scene.add(model);

    auto& clips = result->animations;

    std::cout << "Loaded " << clips.size() << " animation(s):\n";
    for (auto& clip : clips) std::cout << "  - " << clip->name() << "\n";

    auto skeletonHelper = SkeletonHelper::create(*model);
    skeletonHelper->visible = false;
    scene.add(skeletonHelper);

    // Animation mixer — root is the scene returned by GLTF loader
    auto mixer = AnimationMixer(*model);

    // Activate the first clip
    int currentClip = 0;
    auto* currentAction = mixer.clipAction(clips[currentClip]);
    currentAction->play();

    // Crossfade to next animation
    float fadeDuration = 0.5f;
    float clipTimer = 0.f;
    float clipHoldTime = 3.f;

    auto crossfadeToNext = [&] {
        auto* prevAction = currentAction;
        currentClip = (currentClip + 1) % static_cast<int>(clips.size());
        currentAction = mixer.clipAction(clips[currentClip]);
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

    OrbitControls controls{camera, canvas};
    controls.target.set(0, 1, 0);
    controls.update();

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

        mixer.update(dt);
        renderer->render(scene, camera);
    });
}
