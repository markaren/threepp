
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto renderer = GLRenderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setClearColor(Color(Color::aliceblue));
    renderer.setSize(canvas.getSize());

    const auto boxGeometry = SphereGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
    auto box = Mesh::create(boxGeometry, boxMaterial);
    scene->add(box);

    TextureLoader loader{};
    boxMaterial->map = loader.loadTexture("textures/checker.png");

    canvas.onWindowResize([&](WindowSize size){
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    box->rotation.order(Euler::RotationOrders::YZX);
    canvas.animate([&](float dt) {
        box->rotation.y(box->rotation.y() + 0.5f * dt);
        scene->rotation.x(scene->rotation.x() + 1.f * dt);

        renderer.render(scene, camera);
    });
}
