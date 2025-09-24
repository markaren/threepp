
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;


void initFountain(ParticleSystem::Settings& settings);
void initSmoke(ParticleSystem::Settings& settings);
void initFireball(ParticleSystem::Settings& settings);
void initFirework(ParticleSystem::Settings& settings);


int main() {

    Canvas canvas("Particle system", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;

    Scene scene;
    scene.background = Color::gray;
    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 1000);
    camera.position.set(0, 10, 35);

    ParticleSystem engine;

    initFountain(engine.settings());

    engine.initialize();
    scene.add(engine);

    auto grid = GridHelper::create();
    scene.add(grid);

    OrbitControls controls{camera, canvas};

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    int selectedIndex = 0;
    std::vector<std::pair<std::string, std::function<void(ParticleSystem::Settings&)>>> demos{
            {"fountain", initFountain},
            {"smoke", initSmoke},
            {"fireball", initFireball},
            {"firework", initFirework}};

    ImguiFunctionalContext ui(canvas, [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({0, 0}, 0);
        ImGui::Begin("Make selection");
        if (ImGui::BeginCombo("Demos", demos[selectedIndex].first.c_str())) {
            for (int index = 0; index < demos.size(); ++index) {
                const bool isSelected = (selectedIndex == index);
                if (ImGui::Selectable(demos[index].first.c_str(), isSelected)) {
                    selectedIndex = index;

                    demos[index].second(engine.settings());
                    engine.initialize();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::End();
    });


    Clock clock;
    canvas.animate([&]() {
        const auto dt = clock.getDelta();

        engine.update(dt * 0.5f);
        renderer.render(scene, camera);

        ui.render();
    });
}

void initFountain(ParticleSystem::Settings& settings) {

    settings.makeDefault();

    settings.positionStyle = ParticleSystem::Type::BOX;
    settings.positionBase = {0, 0, 0};
    settings.positionSpread = {1, 0, 1};

    settings.velocityStyle = ParticleSystem::Type::BOX;
    settings.velocityBase = {0, 16, 0};
    settings.velocitySpread = {10, 2, 10};

    settings.accelerationBase = {0, -10, 0};

    settings.angleBase = 0;
    settings.angleSpread = 18;
    settings.angleVelocityBase = 0;
    settings.angleVelocitySpread = 36 * 4;

    settings.particlesPerSecond = 200;
    settings.particleDeathAge = 3.0;
    settings.emitterDeathAge = 60;

    settings.setColorTween({0.5, 2}, {{0, 1, 0.5}, {0.8, 1, 0.5}})
            .setOpacityTween({2, 3}, {1, 0})
            .setSizeTween({0, 1}, {0.1, 2});

    TextureLoader tl;
    settings.texture = tl.load(std::string(DATA_FOLDER) + "/textures/star.png");
}

void initSmoke(ParticleSystem::Settings& settings) {

    settings.makeDefault();

    settings.positionStyle = ParticleSystem::Type::BOX;
    settings.positionBase = {0, 0, 0};
    settings.positionSpread = {1, 0, 1};

    settings.velocityStyle = ParticleSystem::Type::BOX;
    settings.velocityBase = {0, 15, 0};
    settings.velocitySpread = {8, 5, 8};
    settings.accelerationBase = {0, -1, 0};

    settings.angleBase = 0;
    settings.angleSpread = 72;
    settings.angleVelocityBase = 0;
    settings.angleVelocitySpread = 72;

    settings.particlesPerSecond = 200;
    settings.particleDeathAge = 2.0;
    settings.emitterDeathAge = 60;

    settings.setColorTween({0.4, 1}, {{0, 0, 0.2}, {0, 0, 0.5}})
            .setOpacityTween({0.8, 2}, {0.5, 0})
            .setSizeTween({0, 1}, {1, 10});

    TextureLoader tl;
    settings.texture = tl.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png");
}

void initFireball(ParticleSystem::Settings& settings) {

    settings.makeDefault();

    settings.positionStyle = ParticleSystem::Type::SPHERE;
    settings.positionBase = {0, 5, 0};
    settings.positionRadius = 2;

    settings.velocityStyle = ParticleSystem::Type::BOX;
    settings.speedBase = 4;
    settings.speedSpread = 0.8;

    settings.particlesPerSecond = 60;
    settings.particleDeathAge = 1.5;
    settings.emitterDeathAge = 60;

    settings.colorBase = {0.02, 1, 0.2};
    settings.blendStyle = Blending::Additive;

    settings.setOpacityTween({0.7, 1}, {1, 0})
            .setSizeTween({0, 1}, {0.1, 15});

    TextureLoader tl;
    settings.texture = tl.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png");
}

void initFirework(ParticleSystem::Settings& settings) {

    settings.makeDefault();

    settings.positionStyle = ParticleSystem::Type::SPHERE;
    settings.positionBase = {0, 10, 0};
    settings.positionRadius = 10;

    settings.velocityStyle = ParticleSystem::Type::SPHERE;
    settings.speedBase = 9;
    settings.speedSpread = 10;

    settings.accelerationBase = {0, -8, 0};

    settings.particlesPerSecond = 3000;
    settings.particleDeathAge = 2.5;
    settings.emitterDeathAge = 0.2;

    settings.colorBase = {0.02, 1, 0.2};
    settings.blendStyle = Blending::Additive;

    settings.setColorTween({0.4, 0.8, 1}, {{0, 1, 1}, {0, 1, 0.6}, {0.8, 1, 0.6}})
            .setOpacityTween({0.2, 0.7, 2.5}, {0.75, 1, 0})
            .setSizeTween({0.3, 0.6, 1.3}, {0.5, 4, 0.1});

    TextureLoader tl;
    settings.texture = tl.load(std::string(DATA_FOLDER) + "/textures/spark.png");
}
