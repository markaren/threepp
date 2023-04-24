
#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

#include "AgxPhysics.hpp"
#include "DrumPulleySystem.hpp"

using namespace threepp;

namespace {

    struct MyKeyListener: KeyListener {

        float speed = 2;

        explicit MyKeyListener(DrumPulleySystem* drum): drum(drum) {}

        void onKeyPressed(KeyEvent evt) override {

            if (evt.key == 266 || evt.key == 329) {// PG_UP
                drum->setSpeed(speed);
            } else if (evt.key == 267 || evt.key == 323) {//PG_DOWN
                drum->setSpeed(-speed);
            }
        }

        void onKeyReleased(KeyEvent evt) override {

            drum->setSpeed(0);
        }

    private:
        DrumPulleySystem* drum;
    };

}// namespace

int main() {

    Canvas canvas("drumPulley", {{"antialiasing", 4}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::blue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(2, 2, 8);

    OrbitControls controls{camera, canvas};
    controls.target = {2, 2, 0};
    controls.update();

    auto light = DirectionalLight::create();
    scene->add(light);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.getAspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    agx::AutoInit init;
    agxSDK::SimulationRef sim = new agxSDK::Simulation();
    sim->setTimeStep(1.0 / 100);
    auto agxVisualisation = AgxVisualisation::create(*sim);
    agxVisualisation->rotateX(-math::PI / 2);
    scene->add(agxVisualisation);

    auto drum = new DrumPulleySystem();
    agxVisualisation->makeVisual(drum);
    agxVisualisation->makeVisual(drum->drum);
    agxVisualisation->makeVisual(drum->weightWire);
    agxVisualisation->makeVisual(drum->mainWire);
    sim->add(drum);

    agxIO::writeFile("drumPulley.agx", sim);

    MyKeyListener listener(drum);
    canvas.addKeyListener(&listener);

    imgui_functional_context ui(canvas.window_ptr(), [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({200, 0}, 0);
        ImGui::Begin("drumPulley");
        ImGui::SliderFloat("speed", &listener.speed, 0.1f, 5.f);
        ImGui::Checkbox("showConstraints", &agxVisualisation->showConstraints);
        controls.enabled = !ImGui::IsWindowHovered();
        ImGui::End();
    });


    canvas.animate([&](float t, float dt) {
        while ((sim->getTimeStamp() + sim->getTimeStep()) < t) {
            sim->stepForward();
        }

        agxVisualisation->updateVisuals();

        renderer.render(scene, camera);
        ui.render();
    });
}
