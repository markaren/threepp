
#include "threepp/objects/ParticleSystem.hpp"
#include "threepp/threepp.hpp"

#if HAS_IMGUI
#include "threepp/extras/imgui/ImguiContext.hpp"
#endif

using namespace threepp;


void initFountain(ParticleSystem& engine);

void initSmoke(ParticleSystem& engine);


int main() {

    Canvas canvas("Particle system", {{"aa", 4}});
    GLRenderer renderer(canvas.size());
    renderer.checkShaderErrors = true;

    Scene scene;
    scene.background = Color::aliceblue;
    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 1000);
    camera.position.set(0, 10, 35);

    ParticleSystem engine;

    initFountain(engine);

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

#if HAS_IMGUI

    int selectedIndex = 0;
    std::vector<std::pair<std::string, std::function<void(ParticleSystem&)>>> demos{
            {"fountain", initFountain},
            {"smoke", initSmoke}};

    ImguiFunctionalContext ui(canvas.windowPtr(), [&] {
        ImGui::SetNextWindowPos({0, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({230, 0}, 0);
        ImGui::Begin("Make selection");
        if (ImGui::BeginCombo("Demos", demos[selectedIndex].first.c_str())) {
            for (int index = 0; index < demos.size(); ++index) {
                const bool isSelected = (selectedIndex == index);
                if (ImGui::Selectable(demos[index].first.c_str(), isSelected)) {
                    selectedIndex = index;

                    demos[index].second(engine);
                    engine.initialize();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::End();
    });
#endif

    Clock clock;
    canvas.animate([&]() {
        float dt = clock.getDelta();

        engine.update(dt * 0.5f);
        renderer.render(scene, camera);

#if HAS_IMGUI
        ui.render();
#endif
    });
}

void initFountain(ParticleSystem& engine) {
    engine.positionStyle = ParticleSystem::Type::BOX;
    engine.positionBase = {0, 0, 0};
    engine.positionSpread = {1, 0, 1};

    engine.positionStyle = ParticleSystem::Type::BOX;
    engine.velocityBase = {0, 16, 0};
    engine.velocitySpread = {10, 2, 10};

    engine.accelerationBase = {0, -10, 0};

    engine.angleBase = 0;
    engine.angleSpread = 18;
    engine.angleVelocityBase = 0;
    engine.angleVelocitySpread = 36 * 4;

    engine.particlesPerSecond = 200;
    engine.particleDeathAge = 3.0;
    engine.emitterDeathAge = 60;

    engine.setColorTween({0.5, 2}, {{0, 1, 0.5}, {0.8, 1, 0.5}})
            .setOpacityTween({2, 3}, {1, 0})
            .setSizeTween({0, 1}, {0.1, 2});

    TextureLoader tl;
    engine.texture = tl.load("data/textures/star.png");
}

void initSmoke(ParticleSystem& engine) {
    engine.positionStyle = ParticleSystem::Type::BOX;
    engine.positionBase = {0, 0, 0};
    engine.positionSpread = {1, 0, 1};

    engine.positionStyle = ParticleSystem::Type::BOX;
    engine.velocityBase = {0, 15, 0};
    engine.velocitySpread = {8, 5, 8};
    engine.accelerationBase = {0, -1, 0};

    engine.angleBase = 0;
    engine.angleSpread = 72;
    engine.angleVelocityBase = 0;
    engine.angleVelocitySpread = 72;

    engine.particlesPerSecond = 200;
    engine.particleDeathAge = 2.0;
    engine.emitterDeathAge = 60;

    engine.setColorTween({0.4, 1}, {{0, 0, 0.2}, {0, 0, 0.5}})
            .setOpacityTween({0.8, 2}, {0.5, 0})
            .setSizeTween({0, 1}, {1, 10});

    TextureLoader tl;
    engine.texture = tl.load("data/textures/smokeparticle.png");
}
