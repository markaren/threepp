// PhysX Vehicle2 playground: the Range Rover Evoque on a course built from
// primitives (ramps, speed bumps, a cone slalom, a crate wall, barrels). The
// car model is the only asset, so the page stays small enough for the browser
// build (WEB in CMakeLists.txt).
//
// W / Up drives, S / Down brakes and then reverses once the car has stopped,
// A / D or Left / Right steer, SPACE is the handbrake, R puts the car back on
// the start line. The panel's buttons do the same while held, for mouse and
// touch.

#include "renderer_factory.hpp"

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include "EvoqueRig.hpp"

#include <PxPhysicsAPI.h>

#include <chrono>
#include <cmath>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    std::shared_ptr<MeshStandardMaterial> material(const Color& color, float roughness = 0.8f) {
        auto m = MeshStandardMaterial::create();
        m->color = color;
        m->roughness = roughness;
        return m;
    }

    struct Body {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* actor;
        PxTransform start;
    };

    // The course. Statics are placed once; the dynamic props are recorded with
    // their start poses so "Reset course" can put them back.
    class Course {
    public:
        Course(Scene& scene, PhysxWorld& world): scene_(scene), world_(world) {
            buildGround();
            buildRamps();
            buildBumps();
            buildSlalom();
            buildCrateWall();
            buildBarrels();
        }

        void reset() {
            for (auto& b : props_) {
                b.actor->setGlobalPose(b.start);
                b.actor->setLinearVelocity(PxVec3(0));
                b.actor->setAngularVelocity(PxVec3(0));
                b.actor->wakeUp();
            }
        }

    private:
        Scene& scene_;
        PhysxWorld& world_;
        std::vector<Body> props_;

        std::shared_ptr<Mesh> addStatic(const std::shared_ptr<BufferGeometry>& geometry,
                                        const std::shared_ptr<Material>& mat,
                                        const Vector3& position, const Euler& rotation = Euler()) {
            auto mesh = Mesh::create(geometry, mat);
            mesh->position.copy(position);
            mesh->rotation.copy(rotation);
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            scene_.add(mesh);
            world_.addStatic(*mesh);
            return mesh;
        }

        void addProp(const std::shared_ptr<Mesh>& mesh, float density, bool convex) {
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            scene_.add(mesh);
            PxRigidDynamic* actor = convex ? world_.addDynamicConvex(*mesh, density)
                                           : world_.add(*mesh, density);
            if (actor) props_.push_back({mesh, actor, actor->getGlobalPose()});
        }

        void buildGround() {
            constexpr float kSize = 300.f;
            auto ground = Mesh::create(BoxGeometry::create(kSize, 1, kSize), material(Color(0x6f7f5a), 1.f));
            ground->position.y = -0.5f;
            ground->receiveShadow = true;
            scene_.add(ground);
            world_.addStatic(*ground);

            auto grid = GridHelper::create(kSize, 150, Color(0x55603f), Color(0x5f6c48));
            grid->position.y = 0.01f;
            scene_.add(grid);

            // Low walls around the edge keep the car on the ground.
            auto wallMat = material(Color(0x8a8f99));
            const float h = kSize / 2;
            addStatic(BoxGeometry::create(kSize, 1.5f, 1), wallMat, {0, 0.75f, h});
            addStatic(BoxGeometry::create(kSize, 1.5f, 1), wallMat, {0, 0.75f, -h});
            addStatic(BoxGeometry::create(1, 1.5f, kSize), wallMat, {h, 0.75f, 0});
            addStatic(BoxGeometry::create(1, 1.5f, kSize), wallMat, {-h, 0.75f, 0});
        }

        // A plate tilted about x with its low edge on the ground, rising toward +z.
        void ramp(const Vector3& lowEdge, float width, float length, float degrees, const Color& color) {
            const float a = math::degToRad(degrees);
            constexpr float kThickness = 0.4f;
            const Vector3 center{lowEdge.x,
                                 0.5f * length * std::sin(a) - 0.5f * kThickness * std::cos(a),
                                 lowEdge.z + 0.5f * length * std::cos(a)};
            addStatic(BoxGeometry::create(width, kThickness, length), material(color), center, Euler(-a, 0, 0));
        }

        void buildRamps() {
            // Straight ahead of the start: a gentle kicker, then a steeper one.
            ramp({0, 0, 30}, 7, 12, 10, Color(0xd9a441));
            ramp({0, 0, 75}, 7, 9, 18, Color(0xd96c41));

            // A table top to the right: up, flat, down.
            const float a = math::degToRad(15.f);
            ramp({30, 0, 20}, 8, 10, 15, Color(0x4f8fd9));
            const float top = 10.f * std::sin(a);
            const float topEnd = 20.f + 10.f * std::cos(a);
            addStatic(BoxGeometry::create(8, top, 12), material(Color(0x4f8fd9)),
                      {30, top / 2, topEnd + 6});
            addStatic(BoxGeometry::create(8, 0.4f, 10), material(Color(0x4f8fd9)),
                      {30, 0.5f * 10.f * std::sin(a) - 0.2f, topEnd + 12 + 0.5f * 10.f * std::cos(a)},
                      Euler(a, 0, 0));
        }

        // Half-buried capsules across the lane.
        void buildBumps() {
            auto mat = material(Color(0xe0d24a));
            for (int i = 0; i < 6; ++i) {
                addStatic(CapsuleGeometry::create(0.25f, 6), mat,
                          {-30, 0.f, 20.f + 5.f * static_cast<float>(i)},
                          Euler(0, 0, math::PI / 2));
            }
        }

        void buildSlalom() {
            auto coneGeometry = ConeGeometry::create(0.3f, 0.8f, 16);
            auto coneMat = material(Color(0xff6a13), 0.6f);
            for (int i = 0; i < 10; ++i) {
                auto cone = Mesh::create(coneGeometry, coneMat);
                cone->position.set(i % 2 == 0 ? -2.f : 2.f, 0.4f, -20.f - 9.f * static_cast<float>(i));
                addProp(cone, 300.f, true);
            }
        }

        void buildCrateWall() {
            auto boxGeometry = BoxGeometry::create(1, 1, 1);
            auto crateMat = material(Color(0xa87a4a));
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 6; ++col) {
                    auto crate = Mesh::create(boxGeometry, crateMat);
                    const float offset = row % 2 == 0 ? 0.f : 0.5f;
                    crate->position.set(-28.f + static_cast<float>(col) * 1.02f + offset,
                                        0.5f + static_cast<float>(row) * 1.01f, -30.f);
                    addProp(crate, 150.f, false);
                }
            }
        }

        void buildBarrels() {
            auto barrelGeometry = CylinderGeometry::create(0.35f, 0.35f, 0.9f, 16);
            auto barrelMat = material(Color(0x2f6fb3), 0.5f);
            for (int i = 0; i < 9; ++i) {
                auto barrel = Mesh::create(barrelGeometry, barrelMat);
                barrel->position.set(28.f + static_cast<float>(i % 3) * 0.8f, 0.45f,
                                     -30.f + static_cast<float>(i / 3) * 0.8f);
                addProp(barrel, 400.f, true);
            }
        }
    };

}// namespace

int main() {

    Canvas canvas("Vehicle playground", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    const Color sky(0xa9c8e8);
    auto scene = Scene::create();
    scene->background = sky;
    scene->fog = Fog(sky, 90, 260);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);

    // The sun's shadow frustum is small and follows the car.
    constexpr float kShadowArea = 25.f;
    const Vector3 sunOffset{-15, 30, -10};
    auto sun = DirectionalLight::create(0xffffff, 2.f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    sun->shadow->bias = -0.0005f;
    auto* shadowCamera = sun->shadow->camera->as<OrthographicCamera>();
    shadowCamera->left = -kShadowArea;
    shadowCamera->right = kShadowArea;
    shadowCamera->top = kShadowArea;
    shadowCamera->bottom = -kShadowArea;
    shadowCamera->nearPlane = 1.f;
    shadowCamera->farPlane = 100.f;
    shadowCamera->updateProjectionMatrix();
    auto sunTarget = Object3D::create();
    sun->setTarget(*sunTarget);
    scene->add(sun);
    scene->add(sunTarget);
    scene->add(HemisphereLight::create(0xdfeeff, 0x5a6448, 0.9f));

    PhysxWorld world;
    Course course(*scene, world);

    auto settings = evoque::vehicleSettings();
    settings.spawnPosition = {0, 1.2f, 0};
    PhysxVehicle vehicle(world, settings);

    evoque::Rig car(world, vehicle, DATA_FOLDER);
    car.body->traverseType<Mesh>([](Mesh& m) { m.castShadow = true; });
    scene->add(car.chassis);

    auto respawn = [&] {
        auto* actor = vehicle.chassisActor();
        actor->setGlobalPose(toPxTransform(settings.spawnPosition, settings.spawnRotation));
        actor->setLinearVelocity(PxVec3(0));
        actor->setAngularVelocity(PxVec3(0));
        actor->wakeUp();
        vehicle.setThrottle(0.f);
        vehicle.setBrake(0.f);
        vehicle.setSteer(0.f);
        vehicle.setGear(PhysxVehicle::Gear::Forward);
    };

    // Keyboard state, and the panel's hold buttons (mouse or touch).
    bool keyForward = false, keyBack = false, keyLeft = false, keyRight = false, keyHandbrake = false;
    bool uiForward = false, uiBack = false, uiLeft = false, uiRight = false;
    bool respawnRequested = false, resetCourseRequested = false;

    auto setKey = [&](Key key, bool down) {
        switch (key) {
            case Key::W:
            case Key::UP: keyForward = down; break;
            case Key::S:
            case Key::DOWN: keyBack = down; break;
            case Key::A:
            case Key::LEFT: keyLeft = down; break;
            case Key::D:
            case Key::RIGHT: keyRight = down; break;
            case Key::SPACE: keyHandbrake = down; break;
            default: break;
        }
    };
    KeyAdapter pressAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::R) respawnRequested = true;
        setKey(evt.key, true);
    });
    KeyAdapter releaseAdapter(KeyAdapter::KEY_RELEASED, [&](KeyEvent evt) { setKey(evt.key, false); });
    canvas.addKeyListener(pressAdapter);
    canvas.addKeyListener(releaseAdapter);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    float steerCmd = 0.f;
    bool braking = false;
    float physicsMs = 0.f;// smoothed cost of world.step + the rig update

    constexpr ImGuiWindowFlags kOverlay = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float s = ui.dpiScale();
        const ImVec2 view = ImGui::GetIO().DisplaySize;
        const float pad = 6 * s;

        ImGui::SetNextWindowPos({pad, pad}, ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("info", nullptr, kOverlay);
        ImGui::TextUnformatted("PhysX Vehicle2");
        ImGui::TextDisabled("W/S drive, A/D steer, SPACE brake");
        ImGui::Text("%5.1f km/h%s", std::abs(vehicle.forwardSpeed()) * 3.6f,
                    vehicle.gear() == PhysxVehicle::Gear::Reverse ? "  reverse" : "");
        ImGui::Text("physics %.2f ms/frame", physicsMs);
        if (ImGui::SmallButton("Back to start (R)")) respawnRequested = true;
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset course")) resetCourseRequested = true;
        ImGui::End();

        // Hold buttons for mouse and touch: active while pressed.
        ImGui::SetNextWindowPos({view.x - pad, view.y - pad}, ImGuiCond_Always, {1, 1});
        ImGui::SetNextWindowBgAlpha(0.3f);
        ImGui::Begin("pedals", nullptr, kOverlay);
        const ImVec2 b{44 * s, 30 * s};
        ImGui::Dummy({b.x, b.y});
        ImGui::SameLine();
        ImGui::Button("Gas", b);
        uiForward = ImGui::IsItemActive();
        ImGui::Button("<", b);
        uiLeft = ImGui::IsItemActive();
        ImGui::SameLine();
        ImGui::Button("Brake", b);
        uiBack = ImGui::IsItemActive();
        ImGui::SameLine();
        ImGui::Button(">", b);
        uiRight = ImGui::IsItemActive();
        ImGui::End();

        ImGui::SetNextWindowPos({pad, view.y - pad}, ImGuiCond_Always, {0, 1});
        ImGui::SetNextWindowSizeConstraints({0, 0}, {view.x * 0.55f, view.y});
        ImGui::SetNextWindowBgAlpha(0.f);
        ImGui::Begin("credit", nullptr, kOverlay | ImGuiWindowFlags_NoInputs);
        ImGui::PushTextWrapPos(view.x * 0.55f - 2 * pad);
        ImGui::TextDisabled("%s", evoque::kCredit);
        ImGui::PopTextWrapPos();
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Vector3 camPos{0, 5, -10};
    Vector3 camTarget{0, 1, 0};

    Clock clock;
    canvas.animate([&] {
        const float dt = std::min(clock.getDelta(), 0.1f);

        if (respawnRequested) {
            respawnRequested = false;
            steerCmd = 0.f;
            respawn();
        }
        if (resetCourseRequested) {
            resetCourseRequested = false;
            course.reset();
        }

        const bool forward = keyForward || uiForward;
        const bool back = keyBack || uiBack;
        const float speed = vehicle.forwardSpeed();

        // S brakes while moving forward and reverses once (nearly) stopped; W does
        // the opposite in reverse.
        if (forward && !back && vehicle.gear() == PhysxVehicle::Gear::Reverse && speed > -0.5f) {
            vehicle.setGear(PhysxVehicle::Gear::Forward);
        } else if (back && !forward && vehicle.gear() == PhysxVehicle::Gear::Forward && speed < 0.5f) {
            vehicle.setGear(PhysxVehicle::Gear::Reverse);
        }
        const bool inReverse = vehicle.gear() == PhysxVehicle::Gear::Reverse;
        const bool accelerate = inReverse ? back : forward;
        braking = (inReverse ? forward : back) || keyHandbrake;

        // Speed-sensitive steering with a slew limit, so the car does not snap
        // round at speed.
        const float steerInput = ((keyLeft || uiLeft) ? 1.f : 0.f) - ((keyRight || uiRight) ? 1.f : 0.f);
        const float steerScale = 1.f / (1.f + std::abs(speed) * 3.6f * 0.015f);
        steerCmd += (steerInput * steerScale - steerCmd) * std::min(1.f, dt * 2.f);

        vehicle.setThrottle(accelerate ? 1.f : 0.f);
        vehicle.setBrake(braking ? 1.f : 0.f);
        vehicle.setSteer(steerCmd);

        const auto t0 = std::chrono::steady_clock::now();
        world.step(dt);
        car.update(vehicle, braking);
        const float stepMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        physicsMs += (stepMs - physicsMs) * 0.05f;

        // Back to the start if the car has left the world.
        if (vehicle.chassisActor()->getGlobalPose().p.y < -20.f) respawn();

        // Chase camera behind the car, smoothed with a fixed time constant. The
        // chassis group carries PhysxWorld's interpolated pose.
        car.chassis->updateMatrixWorld();
        const Matrix4& chassisMat = *car.chassis->matrixWorld;
        Vector3 desiredCam{0, 3.5f, -9.f};
        desiredCam.applyMatrix4(chassisMat);
        desiredCam.y = std::max(desiredCam.y, 1.f);
        Vector3 desiredTarget{0, 1.f, 3.f};
        desiredTarget.applyMatrix4(chassisMat);
        const float follow = 1.f - std::exp(-5.f * dt);
        camPos.lerp(desiredCam, follow);
        camTarget.lerp(desiredTarget, follow);
        camera->position.copy(camPos);
        camera->lookAt(camTarget);

        Vector3 carPos;
        car.chassis->getWorldPosition(carPos);
        sunTarget->position.copy(carPos);
        sun->position.copy(carPos).add(sunOffset);

        renderer->render(*scene, *camera);
        ui.render();
    });
}
