
#include "threepp/objects/Reflector.hpp"
#include "threepp/threepp.hpp"
#include "threepp/utils/URLFetcher.hpp"

#include <iostream>

using namespace threepp;

int main() {

    Canvas canvas("Texture2D", {{"aa", 8}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;

    OrbitControls controls{*camera, canvas};

    TextureLoader tl;

    const auto sphereGeometry = SphereGeometry::create(0.5f, 16, 16);
    const auto sphereMaterial = MeshBasicMaterial::create({{"map", tl.load("data/textures/checker.png")}});
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    sphere->position.setX(1);
    scene->add(sphere);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshBasicMaterial::create();

#ifdef THREEPP_WITH_CURL
    std::string url{"https://raw.githubusercontent.com/mrdoob/three.js/r129/examples/textures/crate.gif"};
    utils::UrlFetcher urlFetcher;
    std::vector<unsigned char> data;
    urlFetcher.fetch(url, data);
    boxMaterial->map = tl.loadFromMemory(url, data);
#else
    boxMaterial->map = tl.load("data/textures/crate.gif");
#endif

    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.setX(-1);
    scene->add(box);

    const auto planeGeometry = PlaneGeometry::create(5, 5);
    const auto planeMaterial = MeshBasicMaterial::create({{"side", Side::Double},
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

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        box->rotation.y += 0.5f * dt;
        sphere->rotation.x += 0.5f * dt;

        renderer.render(*scene, *camera);
    });
}
