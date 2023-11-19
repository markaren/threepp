
#include "threepp/loaders/AssimpLoader.hpp"
#include "threepp/threepp.hpp"
#include <iostream>

using namespace threepp;

int main() {

    Canvas canvas("Morphtargets - sphere", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    Scene scene;

    PerspectiveCamera camera(45, canvas.size().aspect(), 0.2f, 100.f);
    camera.position.set(0, 5, 5);

    auto light1 = PointLight::create(0xff2200, 0.7f);
    light1->position.set(100, 100, 100);
    scene.add(light1);

    auto light2 = PointLight::create(0x22ff00, 0.7f);
    light2->position.set(-100, -100, -100);
    scene.add(light2);

    scene.add(AmbientLight::create(0x111111));

    AssimpLoader loader;
    auto gltf = loader.load("data/models/gltf/AnimatedMorphSphere/AnimatedMorphSphere.gltf");
    scene.add(gltf);

    auto pointsMaterial = PointsMaterial::create();
    pointsMaterial->size = 10;
    pointsMaterial->sizeAttenuation = false;
    pointsMaterial->alphaTest = 0.5;
    pointsMaterial->map = TextureLoader().load("data/textures/sprites/disc.png");
    pointsMaterial->morphTargets = true;

    Mesh* mesh;
    gltf->traverseType<Mesh>([&](Mesh& m) {
        m.rotation.z = math::PI / 2;
        mesh = &m;

        mesh->material()->as<MaterialWithMorphTargets>()->morphTargets = true;

        auto points = Points::create(mesh->shared_geometry(), pointsMaterial);
        points->copyMorphTargetInfluences(&mesh->morphTargetInfluences());
        mesh->add(points);
    });

    OrbitControls controls(camera, canvas);
    controls.minDistance = 0.1;
    controls.maxDistance = 20;

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    float sign = 1;
    float speed = 0.5f;

    Clock clock;
    clock.start();
    canvas.animate([&] {
        auto delta = clock.getDelta();

        auto step = delta * speed;

        mesh->rotation.y += step;

        auto& influence = mesh->morphTargetInfluences()[1];
        influence += step * sign;

        if (influence <= 0 || influence >= 1) {
            sign *= -1;
            influence = std::clamp(influence, 0.01f, 0.99f); // avoid "locking"
        }

        renderer.render(scene, camera);
    });
}
