
#include <iostream>
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/loaders/URDFLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

int main(int argc, char** argv) {

    if (argc != 2) return 1;

    std::string urdfPath = argv[1];

    Canvas canvas{"URDF loader", {{"aa", 4}}};
    GLRenderer renderer(canvas.size());
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 100);
    camera->position.z = 1;

    OrbitControls controls{*camera, canvas};

    auto grid = GridHelper::create();
    scene->add(grid);

    auto light = HemisphereLight::create(Color::aliceblue, Color::grey);
    scene->add(light);

    URDFLoader loader;
    AssimpLoader assimpLoader;
    auto model = loader.load(assimpLoader, urdfPath);
    scene->add(model);

    Box3 bb;
    bb.setFromObject(*model);

    Vector3 size;
    bb.getSize(size);
    camera->position.copy(size).multiplyScalar(1.5).setX(0);
    controls.update();

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        model->getObjectByName("joint_1")->rotation.y += dt * 0.1;
        model->getObjectByName("joint_2")->rotation.z += dt * 0.1;
        model->getObjectByName("joint_3")->rotation.z += dt * 0.1;
        model->getObjectByName("joint_4")->rotation.x += dt * 0.1;
        // model->getObjectByName("joint_5")->rotation.y += dt * 0.1;
        // model->getObjectByName("joint_6")->rotation.y += dt * 0.1;

        model->updateMatrixWorld();

        renderer.render(*scene, *camera);
    });

}