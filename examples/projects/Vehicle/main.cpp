// Drivable PxVehicle2 demo. WASD steer/throttle/brake, R toggles gear,
// SPACE handbrake, Backspace respawns. Vehicle chassis + 4 wheels follow
// the PhysX state via PhysxWorld bindings + per-wheel local poses.

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

using namespace threepp;
using namespace ::physx;

namespace {

    std::shared_ptr<Mesh> makeChassisMesh(const PhysxVehicle::Settings& s) {
        auto mat = MeshPhongMaterial::create();
        mat->color = Color::royalblue;
        auto mesh = Mesh::create(
                BoxGeometry::create(s.chassisWidth, s.chassisHeight, s.chassisLength),
                mat);

        auto roofMat = MeshPhongMaterial::create();
        roofMat->color = Color::lightblue;
        auto roof = Mesh::create(
                BoxGeometry::create(s.chassisWidth * 0.85f, s.chassisHeight * 0.6f, s.chassisLength * 0.45f),
                roofMat);
        roof->position.set(0, s.chassisHeight * 0.5f + s.chassisHeight * 0.3f, -s.chassisLength * 0.05f);
        mesh->add(roof);
        return mesh;
    }

    std::shared_ptr<Mesh> makeWheelMesh(float radius, float halfWidth) {
        auto mat = MeshPhongMaterial::create();
        mat->color = Color::black;
        auto geom = CylinderGeometry::create(radius, radius, halfWidth * 2.f, 24);
        // Cylinder default axis is +Y. Rotate so the symmetry axis aligns with +X (lateral).
        geom->rotateZ(math::PI / 2.f);
        auto mesh = Mesh::create(geom, mat);

        // Spoke marker so wheel rotation is visible.
        auto spokeMat = MeshPhongMaterial::create();
        spokeMat->color = Color::white;
        auto spoke = Mesh::create(
                BoxGeometry::create(halfWidth * 2.05f, radius * 0.15f, radius * 1.7f),
                spokeMat);
        mesh->add(spoke);
        return mesh;
    }

    std::shared_ptr<Mesh> makeRamp(const Vector3& position, float yawDeg) {
        auto mat = MeshLambertMaterial::create();
        mat->color = Color::tan;
        auto mesh = Mesh::create(BoxGeometry::create(6, 0.4f, 4), mat);
        mesh->position.copy(position);
        mesh->rotation.set(math::degToRad(15.f), math::degToRad(yawDeg), 0);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("PhysX Vehicle", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);
    renderer->autoClear = false;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);

    auto sun = DirectionalLight::create(0xffffff, 1.5f);
    sun->position.set(20, 30, 20);
    scene->add(sun);
    scene->add(AmbientLight::create(0xffffff, 0.35f));

    PhysxWorld world;

    auto groundMat = MeshLambertMaterial::create();
    groundMat->color = Color::darkolivegreen;
    auto ground = Mesh::create(BoxGeometry::create(200, 1, 200), groundMat);
    ground->position.y = -0.5f;
    scene->add(ground);
    world.addStatic(*ground);

    // A handful of static ramps + walls to drive over / around.
    std::vector<std::shared_ptr<Mesh>> obstacles;
    obstacles.push_back(makeRamp({15, 0.6f, 5}, 90.f));
    obstacles.push_back(makeRamp({-15, 0.6f, -5}, -90.f));
    obstacles.push_back(makeRamp({0, 0.6f, 25}, 0.f));
    for (auto& ramp : obstacles) {
        scene->add(ramp);
        world.addStatic(*ramp);
    }

    auto wallMat = MeshPhongMaterial::create();
    wallMat->color = Color::saddlebrown;
    for (int i = 0; i < 5; ++i) {
        auto box = Mesh::create(BoxGeometry::create(1, 1, 1), wallMat);
        box->position.set(-6.f + i * 1.05f, 0.5f, -12.f);
        scene->add(box);
        world.add(*box, 50.f);
    }

    PhysxVehicle::Settings settings;
    settings.spawnPosition = {0, 1.2f, 0};

    PhysxVehicle vehicle(world, settings);

    auto chassisMesh = makeChassisMesh(settings);
    scene->add(chassisMesh);
    world.bind(*chassisMesh, *vehicle.chassisActor());

    std::array<std::shared_ptr<Mesh>, 4> wheelMeshes;
    for (int i = 0; i < 4; ++i) {
        wheelMeshes[i] = makeWheelMesh(settings.wheelRadius, settings.wheelHalfWidth);
        chassisMesh->add(wheelMeshes[i]);
    }

    // Input state
    bool throttleDown = false, brakeDown = false, handbrakeDown = false;
    bool steerLeftDown = false, steerRightDown = false;
    bool respawnPressed = false;

    auto keyToggle = [&](Key key, bool down) {
        switch (key) {
            case Key::W:
            case Key::UP: throttleDown = down; break;
            case Key::S:
            case Key::DOWN: brakeDown = down; break;
            case Key::A:
            case Key::LEFT: steerLeftDown = down; break;
            case Key::D:
            case Key::RIGHT: steerRightDown = down; break;
            case Key::SPACE: handbrakeDown = down; break;
            default: break;
        }
    };

    KeyAdapter pressAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::R) {
            vehicle.setGear(vehicle.gear() == PhysxVehicle::Gear::Forward
                                    ? PhysxVehicle::Gear::Reverse
                                    : PhysxVehicle::Gear::Forward);
        }
        if (evt.key == Key::BACKSPACE) respawnPressed = true;
        keyToggle(evt.key, true);
    });
    KeyAdapter releaseAdapter(KeyAdapter::KEY_RELEASED, [&](KeyEvent evt) {
        keyToggle(evt.key, false);
    });
    canvas.addKeyListener(pressAdapter);
    canvas.addKeyListener(releaseAdapter);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    float steerCmd = 0.f, throttleCmd = 0.f, brakeCmd = 0.f;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float w = 280 * ui.dpiScale();
        ImGui::SetNextWindowPos({static_cast<float>(canvas.size().width()) - w, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({w, 0}, 0);
        ImGui::Begin("PhysX Vehicle");
        ImGui::Text("Drive  : W / S");
        ImGui::Text("Steer  : A / D");
        ImGui::Text("Gear   : R (toggle Fwd/Rev)");
        ImGui::Text("Brake  : SPACE (handbrake)");
        ImGui::Text("Respawn: BACKSPACE");
        ImGui::Separator();
        const float speedKmh = vehicle.forwardSpeed() * 3.6f;
        ImGui::Text("Speed   : %.1f km/h", speedKmh);
        const char* gearTxt = vehicle.gear() == PhysxVehicle::Gear::Forward    ? "Forward"
                              : vehicle.gear() == PhysxVehicle::Gear::Reverse  ? "Reverse"
                                                                               : "Neutral";
        ImGui::Text("Gear    : %s", gearTxt);
        ImGui::ProgressBar(throttleCmd, {-1, 0}, "Throttle");
        ImGui::ProgressBar(brakeCmd, {-1, 0}, "Brake");
        ImGui::SliderFloat("Steer", &steerCmd, -1.f, 1.f, "%.2f");
        if (ImGui::Button("Respawn")) respawnPressed = true;
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    // Chase camera state (smoothed)
    Vector3 camPos{0, 4, -10};
    Vector3 camTarget{0, 1, 0};

    constexpr float dt = 1.f / 60.f;

    canvas.animate([&] {

        // Build commands from keyboard.
        const float steerInput = (steerLeftDown ? 1.f : 0.f) - (steerRightDown ? 1.f : 0.f);
        const float steerSlew = std::min(1.f, dt * 4.f);
        steerCmd += (steerInput - steerCmd) * steerSlew;
        throttleCmd = throttleDown ? 1.f : 0.f;
        brakeCmd = (brakeDown || handbrakeDown) ? 1.f : 0.f;

        vehicle.setThrottle(throttleCmd);
        vehicle.setBrake(brakeCmd);
        vehicle.setSteer(steerCmd);

        if (respawnPressed) {
            respawnPressed = false;
            auto* actor = vehicle.chassisActor();
            actor->setGlobalPose(toPxTransform(settings.spawnPosition, settings.spawnRotation));
            actor->setLinearVelocity(PxVec3(0));
            actor->setAngularVelocity(PxVec3(0));
            actor->wakeUp();
            vehicle.setThrottle(0.f);
            vehicle.setBrake(0.f);
            vehicle.setSteer(0.f);
            vehicle.setGear(PhysxVehicle::Gear::Forward);
            steerCmd = 0.f;
        }

        world.step(dt);

        // Drive wheel meshes from vehicle wheel local poses (chassis-space).
        for (int i = 0; i < 4; ++i) {
            const PxTransform wp = vehicle.wheelLocalPose(i);
            wheelMeshes[i]->position.set(wp.p.x, wp.p.y, wp.p.z);
            wheelMeshes[i]->quaternion.set(wp.q.x, wp.q.y, wp.q.z, wp.q.w);
        }

        // Chase camera: lag behind chassis along its local -Z, slightly above.
        const PxTransform pose = vehicle.chassisPose();
        Matrix4 chassisMat;
        chassisMat.compose(
                Vector3(pose.p.x, pose.p.y, pose.p.z),
                Quaternion(pose.q.x, pose.q.y, pose.q.z, pose.q.w),
                Vector3(1, 1, 1));
        Vector3 desiredCam{0, 4.f, -10.f};
        desiredCam.applyMatrix4(chassisMat);
        Vector3 desiredTarget{0, 1.f, 2.f};
        desiredTarget.applyMatrix4(chassisMat);
        const float lerp = std::min(1.f, dt * 5.f);
        camPos.lerp(desiredCam, lerp);
        camTarget.lerp(desiredTarget, lerp);
        camera->position.copy(camPos);
        camera->lookAt(camTarget);

        renderer->clear();
        renderer->render(*scene, *camera);

        ui.render();
    });
}
