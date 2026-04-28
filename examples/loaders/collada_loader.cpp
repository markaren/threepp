
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/loaders/ColladaLoader.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main() {
    Canvas canvas("Collada Demo");
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 5, 10);

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
    stormTrooper->rotateZ(math::PI);
    scene->add(stormTrooper);

    Box3 bb;
    bb.setFromObject(*stormTrooper);

    controls.target = bb.getCenter();
    controls.update();

    std::unique_ptr<AnimationMixer> mixer;
    if (!stormTrooper->animations.empty()) {
        std::cout << "Loaded " << stormTrooper->animations.size() << " animation clip(s)." << std::endl;
        mixer = std::make_unique<AnimationMixer>(*stormTrooper);
        mixer->clipAction(stormTrooper->animations.front())->play();
    }

    auto skeletonHelper = SkeletonHelper::create(*stormTrooper);
    scene->add(skeletonHelper);

    Clock clock;
    canvas.animate([&] {
        if (mixer) mixer->update(clock.getDelta());
        renderer->render(*scene, *camera);
    });

    return 0;
}
