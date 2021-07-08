
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

    auto light = AmbientLight::create(0x00ff00);
    scene->add(light);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshPhongMaterial::create();
    boxMaterial->color.setRGB(1,0,0);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    scene->add(box);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create();
    planeMaterial->color.setHex(Color::yellow);
    planeMaterial->transparent = true;
    planeMaterial->opacity = 0.5f;
    planeMaterial->side = DoubleSide;
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setY(-1);
    plane->rotateX(math::degToRad(90));
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size){
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    box->rotation.order(Euler::RotationOrders::YZX);
    canvas.animate([&](float dt) {
        box->rotation.y(box->rotation.y() + 0.5f * dt);

        renderer.render(scene, camera);
    });
}
