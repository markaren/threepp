
#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

#include "AgxPhysics.hpp"

using namespace threepp;

int main() {

    Canvas canvas("agx_test", {{"antialiasing", 4}});
    GLRenderer renderer(canvas);
    renderer.setClearColor(Color::aliceblue);

    auto scene = Scene::create();
    auto camera = PerspectiveCamera::create(75, canvas.getAspect(), 0.1f, 100);
    camera->position.set(5, 5, 5);

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
    sim->setUniformGravity({0, -9.81, 0});

    agx::RigidBodyRef boxBody = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Box(agx::Vec3{1, 1, 1})));
    boxBody->getMassProperties()->setMass(1);
    boxBody->setPosition({0, 2, 0});
    sim->add(boxBody);

    agx::RigidBodyRef sphereBody = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Sphere(0.5)));
    sphereBody->getMassProperties()->setMass(1000);
    sphereBody->setPosition({-4, 0, 0});
    sim->add(sphereBody);

    agxCollide::GeometryRef planeGeometry = new agxCollide::Geometry(new agxCollide::Box(agx::Vec3{10, 0.5f, 10}));
    planeGeometry->setPosition({0, -0.5, 0});
    sim->add(planeGeometry);

    auto hinge = createConstraint<agx::Hinge>({0, 1, 0}, {0, 0, 1}, boxBody);
    hinge->getMotor1D()->setSpeed(1);
    hinge->getLock1D()->setEnable(false);
    hinge->getMotor1D()->setEnable(true);
    sim->add(hinge);

    auto wire = new agxWire::Wire(0.01, 3);
    wire->add(new agxWire::BodyFixedNode(boxBody, {0, -1, 0}));
    wire->add(new agxWire::BodyFixedNode(sphereBody, {0, 0.5, 0}));
    sim->add(wire);

    agxIO::writeFile("agx_test.agx", sim);

    auto agxVisualisation = AgxVisualisation::create(*sim);
    agxVisualisation->makeVisual(boxBody, MeshPhongMaterial::create({{"color", Color::gray}}));
    agxVisualisation->makeVisual(sphereBody, MeshPhongMaterial::create({{"color", Color::green}}));
    agxVisualisation->makeVisual(planeGeometry, MeshPhongMaterial::create({{"color", Color::gray}}));
    agxVisualisation->makeVisual(hinge);
    agxVisualisation->makeVisual(wire);
    scene->add(agxVisualisation);


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
