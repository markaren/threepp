
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 1;

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setSize(canvas.getSize());

    const auto arrow = ArrowHelper::create({0,1,0}, {0,0,0}, 0.5f, 0xff0000);
    scene->add(arrow);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    arrow->rotation.order(Euler::RotationOrders::YZX);
    canvas.animate([&](float dt) {
        arrow->rotation.y(arrow->rotation.y() + 0.5f * dt);

        renderer.render(scene, camera);
    });
}
