
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 2;

    OrbitControls controls{camera, canvas};

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setSize(canvas.getSize());

    const auto arrow = ArrowHelper::create({0, 1, 0}, {0, 0, 0}, 0.5f, 0xff0000);
    arrow->position.setX(0.5f);
    scene->add(arrow);

    const auto axes = AxesHelper::create(1);
    axes->position.setX(-0.5f);
    scene->add(axes);

    const auto grid = GridHelper::create(5);
    scene->add(grid);

    Box3 box({-1, -1, -1}, {1, 1, 1});
    const auto boxHelper = Box3Helper::create(box);
    boxHelper->position.setY(1);
    scene->add(boxHelper);

    Plane plane(Vector3(0.5, 1, 0.5), 1);
    const auto planeHelper = PlaneHelper::create(plane);
    scene->add(planeHelper);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    arrow->rotation.order(Euler::RotationOrders::ZXY);
    axes->rotation.order(Euler::RotationOrders::YZX);
    float t = 0;
    canvas.animate([&](float dt) {
        arrow->rotation.z(arrow->rotation.z() + 0.5f * dt);
        axes->rotation.y(axes->rotation.y() + 0.5f * dt);

        float sineWave = 0.5f * std::sin(2 * math::PI * 0.1f * t) + 1;
        box.setFromCenterAndSize(
                {0, 0, 0},
                Vector3(1, 1, 1).multiplyScalar(sineWave));

        renderer.render(scene, camera);

        t += dt;
    });
}
