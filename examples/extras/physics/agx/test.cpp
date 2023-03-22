

#include "threepp/threepp.hpp"

#include "threepp/extras/physics/AgxPhysics.hpp"

using namespace threepp;

namespace {

    auto createSphere() {

        const auto geometry = SphereGeometry::create(0.1f);
        const auto material = MeshLambertMaterial::create();
        material->color.setHex(0xff0000);
        auto mesh = Mesh::create(geometry, material);
        return mesh;
    }

    auto createPlane() {

        const auto geometry = PlaneGeometry::create(10, 10);
        geometry->rotateX(-math::PI / 2);
        const auto material = MeshLambertMaterial::create();
        material->color = Color::gray;
        auto mesh = Mesh::create(geometry, material);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 5, 5);

    OrbitControls controls{camera, canvas};

    auto light = HemisphereLight::create();
    scene->add(light);

    auto sphere = createSphere();
    sphere->position.y = 10;
    auto plane = createPlane();

    scene->add(sphere);
    scene->add(plane);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    AgxPhysics agx;
    agx.addMesh(*sphere, 1);
    agx.addMesh(*plane, 0);

    canvas.animate([&](float dt) {
        agx.step(dt);

        renderer.render(scene, camera);
    });
}
