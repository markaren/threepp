// Drivable PxVehicle2 demo. WASD steer/throttle/brake, R toggles gear,
// SPACE handbrake, Backspace respawns. Vehicle chassis + 4 wheels follow
// the PhysX state via PhysxWorld bindings + per-wheel local poses.

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxDebugRenderer.hpp"
#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"

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

}// namespace

int main() {

    Canvas canvas("PhysX Vehicle", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();

    RGBELoader hdrLoader;
    if (auto hdrTexture = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/citrus_orchard_road_puresky_2k.hdr")) {
        scene->background = hdrTexture;
        scene->environment = hdrTexture;
    }

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);

    // Driver POV camera, parented to the chassis so it tracks the body. Will
    // be added to chassisMesh once that exists below. The local pose puts the
    // eyepoint roughly where a driver sits (left side, eye height, just behind
    // the firewall) and faces +Z (PhysX vehicle forward).
    auto povCamera = PerspectiveCamera::create(70, canvas.aspect(), 0.05f, 1000);
    povCamera->position.set(0.4f, 0.4f, -0.1f);
    // Camera default forward is -Z; chassis forward is +Z. Flip so POV faces ahead.
    povCamera->rotation.y = math::PI;

    auto sun = DirectionalLight::create(0xffffff, 1.2f);
    sun->position.set(20, 30, 20);
    scene->add(sun);
    scene->add(AmbientLight::create(0xffffff, 0.1f));

    PhysxWorld world;

    // Fallback ground far below the track to catch the car if it leaves the
    // drivable surface — keeps the demo from spawning the vehicle into the void.
    auto groundMat = MeshLambertMaterial::create();
    groundMat->color = Color::darkolivegreen;
    auto ground = Mesh::create(BoxGeometry::create(500, 1, 500), groundMat);
    ground->position.y = -10.f;
    scene->add(ground);
    world.addStatic(*ground);

    // Race track: visual + trimesh collider per sub-mesh. Native AABB is ~480m
    // across, roughly 2× a real drift course — scale down so the car (real-world
    // meters) reads correctly against it.
    ModelLoader modelLoader;
    auto track = modelLoader.load(std::string(DATA_FOLDER) + "/models/gltf/drift_track/drift_race_track_free.glb");
    track->scale.set(0.5f, 0.5f, 0.5f);
    scene->add(track);

    auto isCone = [](const Mesh& m) {
        return m.name.rfind("Cone", 0) == 0;
    };

    auto isBarrier = [](const Mesh& m) {
        return m.name.rfind("Barrier", 0) == 0;
    };

    auto isBarrierCylinder = [](const Mesh& m) {
        return m.name.rfind("BarrierCylinder", 0) == 0;
    };

    // Static colliders.
    world.addStaticTrimeshTree(*track, [&](const Mesh& m) {
        return m.name.rfind("Rails", 0) == 0
        || m.name.rfind("Road", 0) == 0
        || m.name.rfind("Object", 0) == 0
        || m.name.rfind("Terrain", 0) == 0;
    });

    // Cones become dynamic convex obstacles the car can knock around. Barrier
    // cylinders are tethered to a ground anchor with a D6 joint: linear DOFs
    // locked, angular DOFs free with a spring drive that returns them to
    // upright after the car bumps them.
    track->traverseType<Mesh>([&](Mesh& m) {

        if (isCone(m)) {
            world.addDynamicConvex(m, 5.f);
        } else if (isBarrierCylinder(m)) {
            auto* dyn = world.addDynamicConvex(m, 100.f);
            if (!dyn) return;
            const PxTransform pose = dyn->getGlobalPose();
            auto* anchor = world.physics().createRigidStatic(pose);
            world.scene().addActor(*anchor);

            auto* joint = PxD6JointCreate(
                    world.physics(),
                    anchor, PxTransform(PxIdentity),
                    dyn, PxTransform(PxIdentity));
            // Position locked, rotation free. The spring drive then pulls the
            // body back toward identity orientation.
            joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
            joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
            joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);
            joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
            // Stiffness / damping — tune for desired wobble. Acceleration mode
            // (last arg true) makes the spring feel mass-independent.
            const PxD6JointDrive drive(2000.f, 300.f, PX_MAX_F32, true);
            joint->setDrive(PxD6Drive::eSWING, drive);
            joint->setDrive(PxD6Drive::eTWIST, drive);
            joint->setDrivePosition(PxTransform(PxIdentity));
        } else if (isBarrier(m)) {
            world.addDynamicConvex(m, 500.f);
        }
    });

    PhysxVehicle::Settings settings;
    // Match the Range Rover Evoque visual proportions (model is ~2.1×1.6×4.4 m).
    settings.chassisWidth = 1.95f;
    settings.chassisHeight = 1.4f;
    settings.chassisLength = 4.4f;
    settings.wheelbase = 2.66f;
    settings.trackWidth = 1.65f;
    // 4WD splits drive torque across all four wheels — twice the grip envelope
    // before tires break loose at low speed. The Evoque is AWD anyway.
    settings.drivenWheels = {true, true, true, true};
    // Direct-drive (no gearbox) means torque has to do all the work itself —
    // bump up. Damping was high to fight wheelspin; with 4WD + better grip we
    // can lower it.
    settings.maxThrottleTorque = 1500.f;
    settings.wheelDampingRate = 1.5f;
    // Slightly stickier tires + more longitudinal stiffness so the contact
    // patch transmits force faster than the engine can spin the wheel.
    settings.tireFriction = 2.0f;
    settings.longitudinalStiffness = 100'000.f;
    settings.spawnPosition = {20, 1.2f, -10};
    settings.spawnRotation = Quaternion().setFromAxisAngle({0, 1, 0}, math::degToRad(-90.f));

    PhysxVehicle vehicle(world, settings);

    auto chassisMesh = Group::create();
    scene->add(chassisMesh);
    world.bind(*chassisMesh, *vehicle.chassisActor());
    chassisMesh->add(povCamera);

    auto physxDebug = std::make_shared<PhysxDebugRenderer>(world);
    physxDebug->enableDefaults();
    scene->add(physxDebug);

    // Range Rover Evoque visual rig. The model's native units are roughly mm —
    // scale ×100 for meters. Its bottom (wheel contacts) sits at y=0 in model
    // space; offset down so wheel contacts line up with the rest height of the
    // PhysX chassis (chassis center is wheelRadius+|suspY| above contacts).
    auto carBody = modelLoader.load(
            std::string(DATA_FOLDER) + "/models/gltf/2015_land-rover_range_rover_evoque_coupe/scene.gltf");
    carBody->scale.set(100.f, 100.f, 100.f);
    // Drop the body slightly so its wheel wells line up with the PhysX wheel
    // rigs. Tune by a few tenths if the chassis floats or clips.
    carBody->position.y = -1.f;
    chassisMesh->add(carBody);

    // The .gltf packs AO and metalRoughness into the SAME texture (R=AO,
    // G=rough, B=metal). If R wasn't authored, GL's aoMap reads it as 0
    // (full occlusion) and the body goes dark; WGPU evidently doesn't apply
    // AO from this texture. Drop the aoMap to match.
    // GL doesn't implement KHR_materials_transmission; the gltf's glass material
    // is roughness=0 (perfect mirror) so it reflects the HDR sky and reads as
    // opaque. WGPU handles transmission natively. Swap glass → MeshBasicMaterial
    // for GL only.
    const bool isGL = dynamic_cast<WgpuRenderer*>(renderer.get()) == nullptr;
    carBody->traverseType<Mesh>([isGL](Mesh& m) {
        if (isGL && m.name.rfind("car_range_rover_evoque:glasses_glass_windows_glasses_glass_windows_None", 0) == 0) {
            auto basic = MeshBasicMaterial::create();
            basic->color = Color(0x111122);
            basic->opacity = 0.15f;
            basic->transparent = true;
            basic->depthWrite = false;
            basic->side = Side::Double;
            m.setMaterial(basic);
        }
        if (auto* std = m.material()->as<MeshStandardMaterial>()) {
            std->aoMap = nullptr;
        }
    });

    // Extract the model's wheels into PhysX-driven rigs. The .gltf's per-wheel
    // "pivot" nodes have identity transforms — wheel positions are baked into
    // the geometry — so we compute the hub from the combined AABB and clone
    // each wheel mesh into a rig. The clone's local position and scale recenter
    // the geometry on the hub and convert from native (mm) to meters; the rig
    // itself is driven each frame by vehicle.wheelLocalPose(i).
    //
    // PhysX index → model tag (Settings::drivenWheels).
    const char* wheelTag[4] = {"WheelFL", "WheelFR", "WheelBL", "WheelBR"};
    std::array<std::vector<Mesh*>, 4> wheelParts;
    carBody->traverseType<Mesh>([&](Mesh& m) {
        for (int i = 0; i < 4; ++i) {
            if (m.name.find(wheelTag[i]) != std::string::npos) {
                wheelParts[i].push_back(&m);
                return;
            }
        }
    });

    std::array<std::shared_ptr<Group>, 4> wheelRigs;
    for (int i = 0; i < 4; ++i) {
        wheelRigs[i] = Group::create();
        chassisMesh->add(wheelRigs[i]);

        if (wheelParts[i].empty()) continue;

        // Combined hub center in carBody-local (mm) units.
        Box3 combined;
        combined.makeEmpty();
        for (auto* part : wheelParts[i]) {
            part->geometry()->computeBoundingBox();
            const auto& b = *part->geometry()->boundingBox;
            combined.expandByPoint(b.min());
            combined.expandByPoint(b.max());
        }
        const Vector3 hub = combined.getCenter();

        for (auto* part : wheelParts[i]) {
            // Clone BEFORE hiding the original — Object3D::copy inherits `visible`.
            auto cloned = std::dynamic_pointer_cast<Mesh>(part->clone());
            part->visible = false;// hide the static original
            if (!cloned) continue;
            cloned->visible = true;
            // Rig-local transform that bakes recenter + ×100 (mm→m): a vertex v
            // ends up at 100*(v - hub) in rig space, so the hub sits at the rig
            // origin and the geometry is in meters.
            cloned->position.set(-hub.x, -hub.y, -hub.z);
            wheelRigs[i]->add(cloned);
        }
    }

    // Input state
    bool throttleDown = false, brakeDown = false, handbrakeDown = false;
    bool driverView = false;
    bool steerLeftDown = false, steerRightDown = false;
    bool respawnPressed = false;

    auto keyToggle = [&](Key key, bool down) {
        switch (key) {
            case Key::W:
            case Key::UP:
                throttleDown = down;
                break;
            case Key::S:
            case Key::DOWN:
                brakeDown = down;
                break;
            case Key::A:
            case Key::LEFT:
                steerLeftDown = down;
                break;
            case Key::D:
            case Key::RIGHT:
                steerRightDown = down;
                break;
            case Key::SPACE:
                handbrakeDown = down;
                break;
            default:
                break;
        }
    };

    KeyAdapter pressAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::R) {
            vehicle.setGear(vehicle.gear() == PhysxVehicle::Gear::Forward
                                    ? PhysxVehicle::Gear::Reverse
                                    : PhysxVehicle::Gear::Forward);
        }
        if (evt.key == Key::V) driverView = !driverView;
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

    bool pathTrace = false;
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
        const char* gearTxt = vehicle.gear() == PhysxVehicle::Gear::Forward   ? "Forward"
                              : vehicle.gear() == PhysxVehicle::Gear::Reverse ? "Reverse"
                                                                              : "Neutral";
        ImGui::Text("Gear    : %s", gearTxt);
        ImGui::ProgressBar(throttleCmd, {-1, 0}, "Throttle");
        ImGui::ProgressBar(brakeCmd, {-1, 0}, "Brake");
        ImGui::SliderFloat("Steer", &steerCmd, -1.f, 1.f, "%.2f");
        if (ImGui::Button("Respawn")) respawnPressed = true;
        ImGui::Separator();
        ImGui::Checkbox("PhysX debug", &physxDebug->visible);
        ImGui::Checkbox("Driver view (V)", &driverView);

        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        povCamera->aspect = size.aspect();
        povCamera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    // Chase camera state (smoothed)
    Vector3 camPos{0, 4, -10};
    Vector3 camTarget{0, 1, 0};

    // Orbit overlay: left-drag rotates yaw/pitch, wheel zooms distance. While
    // idle (no drag) the params lerp back toward the default chase (yaw=0 means
    // directly behind the car, matching the original {0, 4, -10} offset).
    constexpr float defaultPitch = 0.38f;// atan(4/10) — height/distance of original chase
    constexpr float defaultDist = 10.77f;// sqrt(4*4 + 10*10)
    float orbitYaw = 0.f;
    float orbitPitch = defaultPitch;
    float orbitDist = defaultDist;
    bool orbiting = false;
    Vector2 lastMousePos;

    struct OrbitMouse: MouseListener {
        float& yaw;
        float& pitch;
        float& dist;
        bool& dragging;
        Vector2& lastPos;
        OrbitMouse(float& y, float& p, float& d, bool& dr, Vector2& lp)
            : yaw(y), pitch(p), dist(d), dragging(dr), lastPos(lp) {}
        void onMouseDown(int button, const Vector2& pos) override {
            if (ImGui::GetIO().WantCaptureMouse) return;
            if (button == 0) {
                dragging = true;
                lastPos = pos;
            }
        }
        void onMouseUp(int button, const Vector2&) override {
            if (button == 0) dragging = false;
        }
        void onMouseMove(const Vector2& pos) override {
            if (!dragging) return;
            const Vector2 d = pos - lastPos;
            lastPos = pos;
            yaw -= d.x * 0.01f;
            pitch = std::clamp(pitch + d.y * 0.01f, 0.05f, 1.4f);
        }
        void onMouseWheel(const Vector2& delta) override {
            if (ImGui::GetIO().WantCaptureMouse) return;
            dist = std::clamp(dist - delta.y, 3.f, 40.f);
        }
    } orbitMouse(orbitYaw, orbitPitch, orbitDist, orbiting, lastMousePos);
    canvas.addMouseListener(orbitMouse);

    // Optional depth-sensor inset (GL only). DepthSensor scans the scene from
    // a forward-facing camera mounted on the chassis, producing a 3D point
    // cloud. The cloud is visualised as Points and rendered into the lower-left
    // viewport from the sensor's POV.
    auto* glRenderer = dynamic_cast<GLRenderer*>(renderer.get());
    std::shared_ptr<DepthSensor> depthSensor;
    std::shared_ptr<Points> depthPoints;
    constexpr unsigned int kSensorW = 256, kSensorH = 192;
    if (glRenderer) {
        depthSensor = std::make_shared<DepthSensor>(60.f, kSensorW, kSensorH, 0.5f, 60.f);
        depthSensor->position.set(0, 0.5f, 2.f);// nose-mounted, forward
        depthSensor->rotation.y = math::degToRad(180.f);
        chassisMesh->add(depthSensor);

        const size_t maxPts = kSensorW * kSensorH;
        auto pcGeom = BufferGeometry::create();
        pcGeom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(maxPts * 3), 3));
        pcGeom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(maxPts * 3), 3));
        pcGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
        pcGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);

        auto pcMat = PointsMaterial::create({{"size", 0.05f}, {"vertexColors", true}});
        depthPoints = Points::create(pcGeom, pcMat);
        depthPoints->frustumCulled = false;
        depthPoints->layers.set(1);// hidden from the main camera; only the inset enables this layer
        scene->add(depthPoints);
    }

    constexpr float dt = 1.f / 60.f;

    canvas.animate([&] {
        // Build commands from keyboard.
        const float steerInput = (steerLeftDown ? 1.f : 0.f) - (steerRightDown ? 1.f : 0.f);
        // Speed-sensitive steering: full lock at standstill, attenuated at speed
        // so the car doesn't snap-spin at highway pace. Real cars do this via EPS.
        const float speedKmhAbs = std::abs(vehicle.forwardSpeed()) * 3.6f;
        const float steerScale = 1.f / (1.f + speedKmhAbs * 0.015f);
        // Slower slew (~0.5s to full lock) — the previous ~0.25s felt twitchy.
        const float steerSlew = std::min(1.f, dt * 2.f);
        steerCmd += (steerInput * steerScale - steerCmd) * steerSlew;
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
        physxDebug->update();

        // Drive wheel rigs from vehicle wheel local poses (chassis-space).
        for (int i = 0; i < 4; ++i) {
            const PxTransform wp = vehicle.wheelLocalPose(i);
            wheelRigs[i]->position.set(wp.p.x, wp.p.y, wp.p.z);
            wheelRigs[i]->quaternion.set(wp.q.x, wp.q.y, wp.q.z, wp.q.w);
        }

        // While idle, fade orbit params back toward the default chase view.
        if (!orbiting) {
            const float fade = std::min(1.f, dt * 1.5f);
            orbitYaw += (0.f - orbitYaw) * fade;
            orbitPitch += (defaultPitch - orbitPitch) * fade;
            orbitDist += (defaultDist - orbitDist) * fade;
        }

        // Build chase/orbit camera in chassis-local space, then transform to world.
        const PxTransform pose = vehicle.chassisPose();
        Matrix4 chassisMat;
        chassisMat.compose(
                Vector3(pose.p.x, pose.p.y, pose.p.z),
                Quaternion(pose.q.x, pose.q.y, pose.q.z, pose.q.w),
                Vector3(1, 1, 1));
        const float cosP = std::cos(orbitPitch);
        Vector3 desiredCam(
                orbitDist * std::sin(orbitYaw) * cosP,
                orbitDist * std::sin(orbitPitch),
                -orbitDist * std::cos(orbitYaw) * cosP);
        desiredCam.applyMatrix4(chassisMat);
        Vector3 desiredTarget{0, 1.f, 2.f};
        desiredTarget.applyMatrix4(chassisMat);
        const float lerp = std::min(1.f, dt * 5.f);
        camPos.lerp(desiredCam, lerp);
        camTarget.lerp(desiredTarget, lerp);
        camera->position.copy(camPos);
        camera->lookAt(camTarget);

        Camera& activeCamera = driverView ? static_cast<Camera&>(*povCamera) : static_cast<Camera&>(*camera);
        renderer->render(*scene, activeCamera);

        // Depth-sensor inset overlay (GL only). Scan from the chassis-mounted
        // sensor, refresh the point cloud, then render an inset from the
        // sensor's POV with layer 1 enabled so the points become visible.
        if (glRenderer) {
            static std::vector<Vector3> cloud;
            cloud.clear();
            depthPoints->visible = false;// hide cloud during scan to avoid feedback
            depthSensor->scan(*renderer, *scene, cloud);
            depthPoints->visible = true;

            auto& geom = *depthPoints->geometry();
            auto* posAttr = geom.getAttribute<float>("position");
            auto* colAttr = geom.getAttribute<float>("color");
            Vector3 sensorWorldPos;
            depthSensor->getWorldPosition(sensorWorldPos);
            const float maxDist = depthSensor->far();
            Color c;
            int idx = 0;
            for (const auto& p : cloud) {
                posAttr->setXYZ(idx, p.x, p.y, p.z);
                const float t = std::min(p.distanceTo(sensorWorldPos) / maxDist, 1.f);
                c.setHSL(0.33f * (1.f - t), 1.f, 0.5f);// near=green, far=red
                colAttr->setXYZ(idx, c.r, c.g, c.b);
                ++idx;
            }
            geom.setDrawRange(0, idx);
            posAttr->needsUpdate();
            colAttr->needsUpdate();

            auto& sensorCam = depthSensor->getCamera();
            sensorCam.layers.enableAll();// see the cloud (on layer 1) plus the world

            const auto canvasSize = canvas.size();
            const int w = canvasSize.width() / 5;
            const int h = canvasSize.height() / 5;
            const int margin = 5;
            Vector4 oldVp;
            glRenderer->getViewport(oldVp);
            glRenderer->setScissorTest(true);
            glRenderer->setViewport(margin, margin, w, h);
            glRenderer->setScissor(margin, margin, w, h);
            glRenderer->render(*scene, sensorCam);
            glRenderer->setScissorTest(false);
            glRenderer->setViewport(oldVp);
        }

        ui.render();
    });
}
