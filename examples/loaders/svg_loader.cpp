#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("SVGLoader", {{"antialiasing", 4}});

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 100;

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto gridHelper = GridHelper::create(160, 10);
    gridHelper->rotation.x = math::PI / 2;
    scene->add(gridHelper);

    SVGLoader loader;
    auto svg = loader.load("data/models/svg/tiger.svg");
    svg->scale.multiplyScalar(0.25f);
    svg->position.x = -70;
    svg->position.y = 70;
    svg->scale.y *= -1;
    scene->add(svg);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    OrbitControls controls{camera, canvas};

    canvas.animate([&]() {
        renderer.render(scene, camera);
    });
}
