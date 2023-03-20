

#include "threepp/threepp.hpp"

#include "threepp/extras/physics/AgxPhysics.hpp"

using namespace threepp;

int main() {

    Canvas canvas;
    GLRenderer renderer(canvas);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.z = 5;

    OrbitControls controls{camera, canvas};

    auto light = AmbientLight::create(Color(0xffffff).multiplyScalar(0.5f));
    scene->add(light);

    const auto boxGeometry = BoxGeometry::create();
    const auto boxMaterial = MeshLambertMaterial::create();
    boxMaterial->color.setHex(0xff0000);
    auto box = Mesh::create(boxGeometry, boxMaterial);
    box->position.x = -1;
    scene->add(box);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    AgxPhysics agx;
    agx.addMesh(*box, 1);

    canvas.animate([&](float dt) {
        agx.step(dt);

        renderer.render(scene, camera);
    });
}
