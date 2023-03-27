
#include "threepp/objects/Reflector.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas(Canvas::Parameters().antialiasing(8));
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    TextureLoader tl;

    const auto sphereGeometry = SphereGeometry::create(0.5f, 16, 16);
    const auto sphereMaterial = MeshBasicMaterial::create({{"map", tl.load("data/textures/checker.png")}});
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(1);
    scene->add(sphere);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();
#ifdef THREEPP_WITH_CURL
    boxMaterial->map = tl.loadFromUrl("https://raw.githubusercontent.com/mrdoob/three.js/r129/examples/textures/crate.gif");
#else
    boxMaterial->map = tl.load("data/textures/crate.gif");
#endif
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setX(-1);
    scene->add(box);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create({{"side", DoubleSide},
                                                          {"map", tl.load("data/textures/brick_bump.jpg")}});
    auto plane = Mesh::create(planeGeometry, planeMaterial);
    plane->position.setZ(-1);
    scene->add(plane);

    Reflector::Options opt;
    opt.clipBias = 0.003f;
    opt.color = 0x777777;
    auto reflector = Reflector::create(PlaneGeometry::create(10, 10), opt);
    reflector->rotateX(math::degToRad(-90));
    reflector->position.setY(-2.5f);
    scene->add(reflector);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    canvas.animate([&](float dt) {
        box->rotation.y += 0.5f * dt;
        sphere->rotation.x += 0.5f * dt;

        renderer.render(scene, camera);
    });
}
