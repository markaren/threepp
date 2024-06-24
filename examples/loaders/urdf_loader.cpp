
#include <iostream>
#include <threepp/loaders/AssimpLoader.hpp>
#include <threepp/loaders/URDFLoader.hpp>
#include <threepp/threepp.hpp>

using namespace threepp;

int main() {

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
    auto model = loader.load(assimpLoader, "C:\\Users\\Lars Ivar Hatledal\\OneDrive - NTNU\\Teaching\\AIS1003\\2023\\Mappe\\Sensur\\10031\\testfiler\\abb-kinetic-devel\\abb_irb2400_support\\urdf\\irb2400.urdf");
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

        model->getObjectByName("joint_1")->rotation.y += dt * 1;

        model->updateMatrixWorld();

        renderer.render(*scene, *camera);
    });

}