
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/threepp.hpp"

#include <iostream>

using namespace threepp;

int main() {
    Canvas canvas("GLTF Demo");
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0, 2, -4);

    OrbitControls controls{*camera, canvas};

    auto ambientLight = AmbientLight::create(0xffffff, 0.2f);
    scene->add(ambientLight);

    auto dirLight = DirectionalLight::create(0xffffff, 1.0f);
    dirLight->position.set(1, 1, -1);
    dirLight->castShadow = true;
    scene->add(dirLight);

    GLTFLoader loader;
    auto result = loader.load(std::string(DATA_FOLDER) + "/models/gltf/Soldier.glb");
    result->scene->traverseType<Mesh>([&](Mesh& mesh) {
        mesh.castShadow = true;
    });

    if (!result) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    scene->add(result->scene);

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

    canvas.animate([&] {
        renderer->render(*scene, *camera);
    });

    return 0;
}
