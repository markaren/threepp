
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Particle system", {{"aa", 4}});
    GLRenderer renderer(canvas.size());

    auto scene = Scene::create();
    scene->background = Color::aliceblue;
    auto camera = PerspectiveCamera::create(75, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 10;

    ParticleSystem engine;

    engine.positionStyle = threepp::ParticleSystem::Type::BOX;
    engine.positionBase = {0, 0, 0};
    engine.positionSpread = {1, 1, 1};

    engine.positionStyle = ParticleSystem::Type::BOX;
    engine.velocityBase = {0, 16, 0};
    engine.velocitySpread = {10, 2, 10};

    engine.accelerationBase = {0, -10, 0};

    engine.angleBase = 0;
    engine.angleSpread = 180;
    engine.angleVelocityBase = 0;
    engine.angleVelocitySpread = 360*4;

    engine.particlesPerSecond = 500;
    engine.particleDeathAge = 3.0;
    engine.emitterDeathAge = 60;

    engine.setColorTween({0.5, 2}, {{0, 1, 0.5}, {0.8, 1, 0.5}})
            .setOpacityTween({2, 3}, {1, 0})
            .setSizeTween({0, 1}, {0.1, 1});

    engine.initialize();
    scene->add(engine);

    OrbitControls controls{*camera, canvas};

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        engine.update(dt);
        renderer.render(*scene, *camera);
    });
}
