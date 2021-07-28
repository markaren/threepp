
#include "threepp/threepp.hpp"
#include "threepp/objects/Reflector.hpp"

using namespace threepp;

int main() {

    Canvas canvas;

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;
    camera->position.y = 2;

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.checkShaderErrors = true;
    renderer.setClearColor(Color(Color::aliceblue));
    renderer.setSize(canvas.getSize());

    const auto sphereGeometry = SphereGeometry::create(0.5f);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x0000ff);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    scene->add(sphere);

    Reflector::Options opt;
    opt.clipBias = 0.003f;
    opt.color = 0x777777;
    auto reflector = Reflector::create(PlaneGeometry::create(10, 10), opt);
    reflector->rotateX(math::degToRad(-90));
    reflector->position.setY(-1);
    scene->add(reflector);

    canvas.onWindowResize([&](WindowSize size) {
      camera->aspect = size.getAspect();
      camera->updateProjectionMatrix();
      renderer.setSize(size);
    });

    sphere->rotation.setOrder(Euler::YZX);
    canvas.animate([&](float dt) {
        sphere->rotation.y += 0.5f * dt;

        renderer.render(scene, camera);
    });

}