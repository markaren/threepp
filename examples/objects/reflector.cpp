
#include "threepp/objects/Reflector.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas{Canvas::Parameters().antialiasing(8)};

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 1000);
    camera->position.z = 5;
    camera->position.y = 2;

    OrbitControls controls{camera, canvas};

    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    const auto sphereGeometry = SphereGeometry::create(0.5f);
    const auto sphereMaterial = MeshBasicMaterial::create();
    sphereMaterial->color.setHex(0x0000ff);
    sphereMaterial->wireframe = true;
    auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
    scene->add(sphere);

    Reflector::Options opt;
    opt.clipBias = 0.003f;
    opt.color = 0x7f7f7f;

    float planeSize{5.f};
    float halfPlaneSize = planeSize / 2;
    {
        auto reflector = Reflector::create(PlaneGeometry::create(planeSize, planeSize), opt);
        reflector->position.setZ(-halfPlaneSize);
        scene->add(reflector);
    }
    {
        auto reflector = Reflector::create(PlaneGeometry::create(planeSize, planeSize), opt);
        reflector->rotateX(math::degToRad(-90));
        reflector->position.setY(-halfPlaneSize);
        scene->add(reflector);
    }
    {
        auto reflector = Reflector::create(PlaneGeometry::create(planeSize, planeSize), opt);
        reflector->rotateY(math::degToRad(-90));
        reflector->position.setX(halfPlaneSize);
        scene->add(reflector);
    }
    {
        auto reflector = Reflector::create(PlaneGeometry::create(planeSize, planeSize), opt);
        reflector->rotateY(math::degToRad(90));
        reflector->position.setX(-halfPlaneSize);
        scene->add(reflector);
    }

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