
#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

#include "AgxPhysics.hpp"

using namespace threepp;

int main() {

    Canvas canvas("agx_test", {{"antialiasing", 4}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::blue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(0, 5, 10);

    OrbitControls controls{camera, canvas};

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

    auto body = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Box(agx::Vec3{1,1,1})));
    sim->add(body);

    auto hinge = createConstraint<agx::Hinge>({0,1,0}, {0,0,1}, body);
    hinge->getMotor1D()->setEnable(true);
    hinge->getMotor1D()->setSpeed(0.1);
    sim->add(hinge);

    agxVisualisation->makeVisual(body);
    agxVisualisation->makeVisual(hinge);

    agxIO::writeFile("agx_test.agx", sim);

    imgui_functional_context ui(canvas.window_ptr(), [&] {
        ImGui::SetNextWindowPos({}, 0, {});
        ImGui::SetNextWindowSize({200, 0}, 0);
        ImGui::Begin("agx_test");
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
