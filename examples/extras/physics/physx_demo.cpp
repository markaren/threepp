
#include "PxEngine.hpp"


using namespace threepp;

int main() {

    Canvas canvas("PhysX");
    GLRenderer renderer(canvas.size());

    Scene scene;
    scene.background = Color::navy;

    PerspectiveCamera camera(60, canvas.aspect());
    camera.position.z = 10;

    auto box = Mesh::create(BoxGeometry::create(1,1,1), MeshStandardMaterial::create());
    box->position.y = 10;
    scene.add(box);

    auto sphere = Mesh::create(SphereGeometry::create(0.5), MeshStandardMaterial::create({{"color", Color::green}}));
    sphere->position.y = 12;
    scene.add(sphere);

    auto ground = Mesh::create(BoxGeometry::create(10, 0.1, 10), MeshStandardMaterial::create({{"color", Color::blueviolet}}));
    scene.add(ground);

    auto light = HemisphereLight::create(Color::brown, Color::blanchedalmond);
    light->position.y = 5;
    scene.add(light);

    PxEngine engine;
    engine.registerMeshDynamic(*sphere);
    engine.registerMeshDynamic(*box);
    engine.registerMeshStatic(*ground);

    OrbitControls controls(camera, canvas);

    Clock clock;
    canvas.animate([&] {
        auto dt = clock.getDelta();

        renderer.render(scene, camera);

        engine.step(dt);
    });
}
