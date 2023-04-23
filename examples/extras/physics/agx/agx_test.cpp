
#include "threepp/extras/imgui/imgui_context.hpp"
#include "threepp/threepp.hpp"

#include "AgxPhysics.hpp"

using namespace threepp;

agxSDK::AssemblyRef drumPulley(AgxVisualisation& vis) {

    auto assembly = new agxSDK::Assembly();

    auto pulleyPosition = agx::Vec3(3, 0, 5);
    auto blockPosition = agx::Vec3(2, 0, 2);
    auto blockRadius = 0.1;// m
    auto weightPos = agx::Vec3(4, 0, 0);
    auto weightSize = agx::Vec3(0.5);
    auto hookPosition = weightPos + agx::Vec3(0, 0, 2);
    auto mainWireYoungsModulusStretch = 60E9;  // GPa, resisting to stretch
    auto mainWireYoungsModulusBend = 6E9;      // GPa, resisting to bend
    auto weightWireYoungsModulusStretch = 30E9;// GPa, resisting to stretch
    auto weightWireYoungsModulusBend = 6E9;    // GPa, resisting to bend
    auto hookMass = 15;                        // kg
    auto weightWireRadius = 0.011;             // m
    auto mainWireRadius = 0.011;               // m
    auto wireResolutionPerUnitLength = 3;      // number of lumped elements per meter
    auto weightMass = 9000;                    // kg

    auto drumMotorTorque = 1E5;// Nm
    auto drumBrakeTorque = 1E5;// Nm

    // Create a weight
    auto weightGeometry = new agxCollide::Geometry(new agxCollide::Box(weightSize));
    auto weightBody = new agx::RigidBody();
    weightBody->add(weightGeometry);
    weightBody->getMassProperties()->setMass(weightMass);
    weightBody->setPosition(weightPos);

    assembly->add(weightBody);

    // Create a hook and give it a mass
    auto hookBody = new agx::RigidBody(new agxCollide::Geometry(new agxCollide::Sphere(0.06)));
    hookBody->setPosition(hookPosition);
    hookBody->getMassProperties()->setMass(hookMass);
    assembly->add(hookBody);

    auto weightWire = new agxWire::Wire(weightWireRadius, wireResolutionPerUnitLength);

    auto wireMaterial = new agx::Material("WeightWireMaterial");
    wireMaterial->getBulkMaterial()->setDensity(1100);
    wireMaterial->getWireMaterial()->setYoungsModulusStretch(weightWireYoungsModulusStretch);
    wireMaterial->getWireMaterial()->setYoungsModulusBend(weightWireYoungsModulusBend);
    wireMaterial->getBulkMaterial()->setYoungsModulus(200E9);

    weightWire->setMaterial(wireMaterial);

    auto weightNode1 = new agxWire::BodyFixedNode(weightBody, agx::Vec3(-weightSize.x(), 0, weightSize.z()));
    auto weightNode2 = new agxWire::BodyFixedNode(weightBody, agx::Vec3(weightSize.x(), 0, weightSize.z()));
    auto hookEye = new agxWire::EyeNode(hookBody, agx::Vec3(0, 0, -0.06));
    hookEye->getMaterial()->setFrictionCoefficient(0.5);

    weightWire->add(weightNode1);
    weightWire->add(hookEye);
    weightWire->add(weightNode2);

    assembly->add(weightWire);

    auto drumBody = new agx::RigidBody();
    drumBody->getMassProperties()->setMass(1000);
    drumBody->setPosition(0, 0, 0);
    assembly->add(drumBody);
    assembly->add(new agx::LockJoint(drumBody));// Lock this body to the world.

    auto drumAttachment = new agx::Frame();
    drumAttachment->setLocalRotate(agx::EulerAngles(math::PI / 2, math::degToRad(190), 0));
    drumBody->addAttachment(drumAttachment, "DrumAttachment");

    auto kernelRadius = 0.25;
    auto drumLength = 2;

    auto drum = new agxWire::WireSimpleDrumController(kernelRadius, drumLength, 5, false);
    drum->setAutoFeed(true);
    drum->setSpeed(0);
    drum->setForceRange(agx::RangeReal(0, 0));
    drum->setBrakeForceRange(agx::RangeReal(-drumBrakeTorque, drumMotorTorque));

    auto mainWire = new agxWire::Wire(mainWireRadius, wireResolutionPerUnitLength);

    auto wireMaterial2 = new agx::Material("MainWireMaterial");
    wireMaterial2->getBulkMaterial()->setDensity(2200);
    wireMaterial2->getWireMaterial()->setYoungsModulusStretch(mainWireYoungsModulusStretch);
    wireMaterial2->getWireMaterial()->setYoungsModulusBend(mainWireYoungsModulusBend);


    // Create a fixed block that will collide with the wire
    auto blockGeom = new agxCollide::Geometry(new agxCollide::Cylinder(blockRadius, 1.5));
    blockGeom->setPosition(blockPosition);
    assembly->add(blockGeom);

    // Create a pulley with a hinge.
    auto pulleyBody = new agx::RigidBody();
    pulleyBody->setPosition(pulleyPosition);
    pulleyBody->add(new agxCollide::Geometry(new agxCollide::Cylinder(0.075, 0.3)));
    pulleyBody->getMassProperties()->setMass(15);

    // Create the geometry for the pulley
    auto m = agx::AffineMatrix4x4::translate(agx::Vec3(0, -0.15, 0));
    pulleyBody->add(new agxCollide::Geometry(new agxCollide::Cylinder(0.15, 0.15)), m);

    m = agx::AffineMatrix4x4::translate(agx::Vec3(0, 0.15, 0));
    pulleyBody->add(new agxCollide::Geometry(new agxCollide::Cylinder(0.15, 0.15)), m);

    assembly->add(pulleyBody);

    // Create a hinge and attach the pulley to it, (other side of constraint is attached to the world)
    auto hingeFrame = new agx::Frame();
    hingeFrame->setLocalRotate(agx::EulerAngles(math::PI / 2, 0, 0));
    auto hinge = new agx::Hinge(pulleyBody, hingeFrame);

    // To simulate the "internal" friction on the pulley, we enable the motor, set speed to zero,
    // then we tune the friction with compliance. We could set min/max torque. But having a large system with many many
    // small force ranges will might things complicated for the solver.
    hinge->getMotor1D()->setEnable(true);
    hinge->getMotor1D()->setSpeed(0);
    hinge->getMotor1D()->setCompliance(1E-1);
    assembly->add(hinge);

    // Rotate the drum so that the attachment point comes on top (default is below the drum)
    drum->setRotation(agx::EulerAngles(0, math::degToRad(190), math::PI));

    // Now route the main wire
    // Create a FreeNode (positioned in world coordinates), the routed line will pass this point
    // First route the wire to the drum
    auto lengthOfWire = 12;
    mainWire->add(drum, lengthOfWire);

    // a FreeNode is positioned in world coordinates
    mainWire->add(new agxWire::FreeNode(blockPosition + agx::Vec3((blockRadius + mainWire->getRadius()) / std::sqrt(2), 0,
                                                                  -(blockRadius + mainWire->getRadius()) / std::sqrt(2))));
    mainWire->add(new agxWire::FreeNode(pulleyPosition + agx::Vec3(-0.1, 0, 0.1)));
    mainWire->add(new agxWire::FreeNode(pulleyPosition + agx::Vec3(0.1, 0, 0.1)));

    // And end at the hook, where we attach the wire with a BodyFixedNode(coords given relative to the body)
    auto weightNode = new agxWire::ConnectingNode(hookBody, agx::Vec3(0, 0, 0.06), 0);
    mainWire->add(weightNode);
    mainWire->add(weightNode->getCmNode());

    // And last(important) we add the wire to the simulation
    assembly->add(mainWire);

    // Now attach the drum to the drumBody using the attachment specified earlier
    drum->attach(drumBody, drumAttachment);

    vis.makeVisual(mainWire);
    vis.makeVisual(weightWire);

    vis.makeVisual(pulleyBody);
    vis.makeVisual(weightBody);
    vis.makeVisual(drum);
    vis.makeVisual(hookBody);

    vis.makeVisual(blockGeom);

    vis.makeVisual(hinge);

    return assembly;
}

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

    auto drum = drumPulley(*agxVisualisation);
    sim->add(drum);

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
