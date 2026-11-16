// Play a trained Boston Dynamics Spot policy natively in C++, with no Python and
// no torch, on a landscape built for it: a meadow, a granite stair cut into a
// dry-stone terrace, a rubble patch and mossy boulders under an HDR sky. It
// runs on the desktop and in the browser (WEB in CMakeLists.txt).
//
// The policy is spot_policy.tpnn beside this file, run by SpotPolicy; the robot
// and the observation contract are SpotScene.hpp; the landscape is
// SpotCourse.hpp. The robot is threepp_data's urdf/spot, which carries its own
// Boston Dynamics licence (see the NOTICE there).
//
// The policy acts every tenth 2 ms physics substep from inside PhysxWorld's step
// loop, and each frame advances the world by the frame's wall-clock time. Spot
// therefore walks at real-time speed on any display, and its visuals are
// interpolated between substeps.
//
//   spot_play                 # interactive window
//   spot_play --vulkan        # interactive window, Vulkan (if built with it)
//   spot_play --policy P.tpnn # use a specific exported policy
//   spot_play --urdf R.urdf   # use a specific robot description
//   spot_play --data DIR      # read the assets from another threepp_data checkout
//   spot_play --check 2500    # headless: up the stair, through the rubble, down the slope
//
// DRIVE (body frame): W / S or Up / Down walk, A / D or Left / Right turn,
// Q / E sidestep, R back to the start, H shows or hides the height scan.
// Drag to look around, scroll to zoom.

#include "SpotCourse.hpp"
#include "SpotScene.hpp"
#include "renderer_factory.hpp"

#include "threepp/threepp.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/loaders/RGBELoader.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

using namespace threepp;
using namespace spot;

namespace {

    std::string defaultPolicyPath() {
#ifdef PROJECT_FOLDER
        return std::string(PROJECT_FOLDER) + "/examples/projects/Spot/spot_policy.tpnn";
#else
        return "spot_policy.tpnn";
#endif
    }

#ifdef DATA_FOLDER
    std::string dataDir = DATA_FOLDER;
#else
    std::string dataDir = "data";
#endif
    std::string dataFolder() { return dataDir; }

    // The robot ships in threepp_data (urdf/spot), fetched by the build: nothing is
    // downloaded at run time.
    std::string defaultUrdfPath() { return dataFolder() + "/urdf/spot/spot_physics.urdf"; }

    PhysxWorld::Settings spotWorldSettings() {
        PhysxWorld::Settings s;
        s.gravity = Vector3(0, 0, -9.81f);// Z-up, as Isaac / the Python deploy
        s.fixedTimestep = SUBSTEP_DT;     // the policy acts every DECIMATION substeps
        s.maxSubSteps = 40;               // a frame may cover up to 80 ms of simulation
        return s;
    }

    float yawOf(const std::array<float, 7>& rs) {
        float R[3][3];
        quatToR(rs[3], rs[4], rs[5], rs[6], R);
        return std::atan2(R[1][0], R[0][0]);
    }

    float wrapAngle(float a) {
        a = std::fmod(a + math::PI, 2.f * math::PI);
        if (a < 0) a += 2.f * math::PI;
        return a - math::PI;
    }

    // Turns held keys into the policy's velocity command, once per control tick: an
    // acceleration limit on the walk, and a light heading hold while not turning (the
    // policy regulates yaw RATE to zero, so any bias would slowly spiral).
    struct Pilot {
        bool forward = false, back = false, turnLeft = false, turnRight = false, stepLeft = false, stepRight = false;

        SpotController::Command command(const Articulation& art, bool standMode) {
            const float vxKey = (forward ? 1.0f : 0.f) - (back ? 0.6f : 0.f);
            const float vyKey = (stepLeft ? 0.6f : 0.f) - (stepRight ? 0.6f : 0.f);
            const float turn = (turnLeft ? 1.2f : 0.f) - (turnRight ? 1.2f : 0.f);
            // At most 3 m/s^2. The clamp lands exactly on the target, so releasing the
            // keys still reaches an exact zero and a stand-mode policy stands.
            cmd_[0] += std::clamp(vxKey - cmd_[0], -0.06f, 0.06f);
            cmd_[1] += std::clamp(vyKey - cmd_[1], -0.06f, 0.06f);
            const float vx = cmd_[0], vy = cmd_[1];

            const float yaw = yawOf(art.rootState());
            float wz;
            if (turn != 0.f || (standMode && vx == 0.f && vy == 0.f)) {
                wz = turn;
                heading_ = yaw;
                locked_ = true;
            } else {
                if (!locked_) {
                    heading_ = yaw;
                    locked_ = true;
                }
                wz = std::clamp(-2.0f * wrapAngle(yaw - heading_), -1.0f, 1.0f);
            }
            return {vx, vy, wz};
        }

        void reset() {
            cmd_ = {0.f, 0.f};
            locked_ = false;
        }

    private:
        std::array<float, 2> cmd_{0.f, 0.f};
        float heading_ = 0.f;
        bool locked_ = false;
    };

    float uprightness(const Articulation& art) {
        const auto rs = art.rootState();
        float R[3][3];
        quatToR(rs[3], rs[4], rs[5], rs[6], R);
        return R[2][2];// body local-Z . world-Z
    }

    // Headless: stand, then walk straight up the stair, through the rubble patch
    // on the terrace and down its east slope, on the same course and control path
    // as the window.
    int runCheck(const SpotPolicy& policy, const std::string& urdf, int maxTicks) {
        PhysxWorld world(spotWorldSettings());
        course::Course land(world, nullptr, dataFolder(), false);
        SpotRobot spot = loadSpot(world, urdf);
        SpotController ctrl(spot, policy, [&](float x, float y) { return land.heightAt(x, y); });
        Pilot pilot;
        ctrl.setCommandSource([&] { return pilot.command(*spot.art, policy.standMode()); });
        ctrl.attach(world);

        const auto sp = course::Course::spawn();
        ctrl.spawnStance(sp.x, sp.y, land.heightAt(sp.x, sp.y), 0.f, 60);
        int tick = 0;
        while (ctrl.holdTicksLeft() > 0) {
            world.step(CONTROL_DT);
            ++tick;
        }

        const auto s0 = spot.art->rootState();
        float minUp = 1.f;
        for (int i = 0; i < 100; ++i, ++tick) {
            world.step(CONTROL_DT);
            minUp = std::min(minUp, uprightness(*spot.art));
        }
        const auto s1 = spot.art->rootState();
        std::cout << "[check] stand 100 ticks" << (policy.standMode() ? " (stand mode)" : "") << ": drift "
                  << std::hypot(s1[0] - s0[0], s1[1] - s0[1]) << " m\n";

        pilot.forward = true;
        int climbedAt = -1, rubbleAt = -1, downAt = -1;
        const auto t0 = std::chrono::steady_clock::now();
        int walked = 0;
        for (; walked < maxTicks; ++walked) {
            world.step(CONTROL_DT);
            const auto rs = spot.art->rootState();
            minUp = std::min(minUp, uprightness(*spot.art));
            if (climbedAt < 0 && rs[0] > course::STAIR_BACK + 0.5f && rs[2] > course::TERRACE_H + 0.3f) {
                climbedAt = walked;
                std::cout << "[check] up the stair after " << walked * CONTROL_DT << " s\n";
            }
            if (rubbleAt < 0 && rs[0] > course::RUBBLE_X1 + 0.6f) {
                rubbleAt = walked;
                std::cout << "[check] through the rubble after " << walked * CONTROL_DT << " s\n";
            }
            if (downAt < 0 && rs[0] > course::SLOPE_END + 1.f) {
                downAt = walked;
                std::cout << "[check] down the east slope after " << walked * CONTROL_DT << " s\n";
                break;
            }
            if (minUp < 0.5f) break;
        }
        const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const auto rs = spot.art->rootState();
        std::cout << "[check] base=(" << rs[0] << "," << rs[1] << "," << rs[2] << ")  min upZ=" << minUp
                  << "  " << ms / std::max(walked, 1) << " ms per control tick\n";
        const bool ok = climbedAt >= 0 && rubbleAt >= 0 && downAt >= 0 && minUp > 0.5f;
        std::cout << (ok ? "SPOT C++ PLAY CHECK: OK\n" : "SPOT C++ PLAY CHECK: FAIL\n");
        return ok ? 0 : 1;
    }

    int runInteractive(const SpotPolicy& policy, const std::string& urdf) {

        Canvas canvas(Canvas::Parameters().title("threepp - Spot").size(1280, 720).antialiasing(4).vsync(true));
        auto renderer = createRenderer(canvas);
        renderer->shadowMap().enabled = true;
        renderer->toneMapping = ToneMapping::ACESFilmic;
        renderer->toneMappingExposure = 0.85f;

        auto scene = Scene::create();
        // The sky is an equirect HDR authored Y-up; a quarter turn about x stands it
        // up in this Z-up world, for the backdrop and the image-based light alike, and
        // a turn about its own up axis puts its sun to the left of the start's view and
        // a little behind: the stair ahead is lit from the side, so its treads catch the
        // sun and its risers do not, instead of the two reading as one flat face.
        constexpr float skyYaw = math::PI - 0.66f;
        if (auto sky = RGBELoader().load(dataFolder() + "/textures/env/kloofendal_48d_partly_cloudy_puresky_2k.hdr")) {
            scene->background = sky;
            scene->environment = sky;
            scene->backgroundRotation.set(math::PI / 2, skyYaw, 0);
            scene->environmentRotation.set(math::PI / 2, skyYaw, 0);
        }
        scene->fog = FogExp2(Color(0.66f, 0.72f, 0.80f), 0.006f);

        // The sun: the HDR's own disc is clamped out of its image light and this
        // light takes its place (EnvSunPolicy::Auto), re-aimed at the measured disc
        // once the renderer has found it. Its shadow box follows Spot.
        auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.9f), 3.0f);
        Vector3 sunDir(-0.2f, 0.65f, 0.74f);// until the disc is measured
        sunDir.normalize();
        sun->castShadow = true;
        sun->shadow->mapSize.set(2048, 2048);
        sun->shadow->bias = -0.0004f;
        sun->shadow->normalBias = 0.02f;
        if (auto* sc = sun->shadow->camera->as<OrthographicCamera>()) {
            sc->left = -11.f;
            sc->right = 11.f;
            sc->top = 11.f;
            sc->bottom = -11.f;
            sc->nearPlane = 0.5f;
            sc->farPlane = 80.f;
            sc->updateProjectionMatrix();
        }
        auto sunTarget = Object3D::create();
        sun->setTarget(*sunTarget);
        scene->add(sun);
        scene->add(sunTarget);

        PhysxWorld world(spotWorldSettings());
        course::Course land(world, scene.get(), dataFolder());

        SpotRobot spot = loadSpot(world, urdf);// the URDF's own <visual> meshes render
        for (auto& m : spot.meshes) scene->add(m);
        SpotController ctrl(spot, policy, [&](float x, float y) { return land.heightAt(x, y); });
        Pilot pilot;
        ctrl.setCommandSource([&] { return pilot.command(*spot.art, policy.standMode()); });
        ctrl.attach(world);

        const auto respawn = [&] {
            const auto sp = course::Course::spawn();
            ctrl.spawnStance(sp.x, sp.y, land.heightAt(sp.x, sp.y), 0.f, 40);
            pilot.reset();
        };
        respawn();

        // What the policy reads: the 45 points of its height scan.
        auto scanDots = InstancedMesh::create(SphereGeometry::create(0.022f, 8, 6),
                                              MeshBasicMaterial::create({{"color", Color(0x7ff0ff)}}), N_SCAN);
        scanDots->frustumCulled = false;
        scene->add(scanDots);
        bool showScan = true;

        auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.05f, 600.f);
        camera->up.set(0, 0, 1);
        camera->position.set(-3.4f, -1.4f, 1.6f);
        OrbitControls controls(*camera, canvas);
        controls.enablePan = false;
        controls.enableKeys = false;// the arrows drive Spot
        controls.enableDamping = true;
        controls.dampingFactor = 0.12f;
        controls.minDistance = 1.2f;
        controls.maxDistance = 14.f;
        controls.maxPolarAngle = math::degToRad(86.f);
        controls.target.set(0, 0, 0.5f);
        controls.update();

        // Keys, and the panel's hold buttons for mouse and touch.
        struct Held {
            bool fwd = false, back = false, left = false, right = false, sl = false, sr = false;
        } keys, ui;
        bool respawnRequested = false;
        auto setKey = [&](Key key, bool down) {
            switch (key) {
                case Key::W:
                case Key::UP: keys.fwd = down; break;
                case Key::S:
                case Key::DOWN: keys.back = down; break;
                case Key::A:
                case Key::LEFT: keys.left = down; break;
                case Key::D:
                case Key::RIGHT: keys.right = down; break;
                case Key::Q: keys.sl = down; break;
                case Key::E: keys.sr = down; break;
                default: break;
            }
        };
        KeyAdapter pressAdapter(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
            if (evt.key == Key::R) respawnRequested = true;
            if (evt.key == Key::H) showScan = !showScan;
            setKey(evt.key, true);
        });
        KeyAdapter releaseAdapter(KeyAdapter::KEY_RELEASED, [&](KeyEvent evt) { setKey(evt.key, false); });
        canvas.addKeyListener(pressAdapter);
        canvas.addKeyListener(releaseAdapter);

        IOCapture capture{};
        capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
        canvas.setIOCapture(&capture);

        float physicsMs = 0.f;// smoothed cost of world.step (physics + policy)
        float speed = 0.f;
        bool fallen = false;

        constexpr ImGuiWindowFlags kOverlay = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                              ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;
        ImguiFunctionalContext gui(canvas, *renderer, [&] {
            const float s = gui.dpiScale();
            const ImVec2 view = ImGui::GetIO().DisplaySize;
            const float pad = 8 * s;

            ImGui::SetNextWindowPos({pad, pad}, ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.55f);
            ImGui::Begin("info", nullptr, kOverlay);
            ImGui::TextUnformatted("Spot, a learned walking policy on PhysX");
            ImGui::TextDisabled("W/S walk  A/D turn  Q/E sidestep  R restart");
            ImGui::TextDisabled("drag to look around, scroll to zoom");
            ImGui::Text("%4.2f m/s   physics + policy %.2f ms/frame", speed, physicsMs);
            ImGui::Checkbox("Height scan (H)", &showScan);
            ImGui::SameLine();
            if (ImGui::SmallButton("Restart (R)")) respawnRequested = true;
            if (fallen) ImGui::TextColored({1.f, 0.75f, 0.4f, 1.f}, "Spot is down: press R");
            ImGui::End();

            ImGui::SetNextWindowPos({view.x - pad, view.y - pad}, ImGuiCond_Always, {1, 1});
            ImGui::SetNextWindowBgAlpha(0.3f);
            ImGui::Begin("drive", nullptr, kOverlay);
            const ImVec2 b{46 * s, 34 * s};
            ImGui::Button("<<", b);
            ui.sl = ImGui::IsItemActive();
            ImGui::SameLine();
            ImGui::Button("Walk", b);
            ui.fwd = ImGui::IsItemActive();
            ImGui::SameLine();
            ImGui::Button(">>", b);
            ui.sr = ImGui::IsItemActive();
            ImGui::Button("Left", b);
            ui.left = ImGui::IsItemActive();
            ImGui::SameLine();
            ImGui::Button("Back", b);
            ui.back = ImGui::IsItemActive();
            ImGui::SameLine();
            ImGui::Button("Right", b);
            ui.right = ImGui::IsItemActive();
            ImGui::End();

            ImGui::SetNextWindowPos({pad, view.y - pad}, ImGuiCond_Always, {0, 1});
            ImGui::SetNextWindowSizeConstraints({0, 0}, {view.x * 0.6f, view.y});
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGui::Begin("credit", nullptr, kOverlay | ImGuiWindowFlags_NoInputs);
            ImGui::PushTextWrapPos(view.x * 0.6f - 2 * pad);
            ImGui::TextUnformatted("Spot model: Boston Dynamics SDK, under its own licence (see its NOTICE). "
                                   "Ground, stone, rocks and sky: Poly Haven, CC0.");
            ImGui::PopTextWrapPos();
            ImGui::End();
        });

        canvas.onWindowResize([&](WindowSize size) {
            camera->aspect = size.aspect();
            camera->updateProjectionMatrix();
            renderer->setSize(size);
        });

        Object3D& base = *spot.meshes.front();// the root link's collider, interpolated by PhysxWorld
        Vector3 follow;
        base.getWorldPosition(follow);
        float camYaw = yawOf(spot.art->rootState());
        bool sunAligned = false;
        float downFor = 0.f;
        Matrix4 m;

        Clock clock;
        canvas.animate([&] {
            const float dt = std::min(clock.getDelta(), 0.1f);

            if (respawnRequested) {
                respawnRequested = false;
                respawn();
                downFor = 0.f;
            }
            pilot.forward = keys.fwd || ui.fwd;
            pilot.back = keys.back || ui.back;
            pilot.turnLeft = keys.left || ui.left;
            pilot.turnRight = keys.right || ui.right;
            pilot.stepLeft = keys.sl || ui.sl;
            pilot.stepRight = keys.sr || ui.sr;

            const auto t0 = std::chrono::steady_clock::now();
            world.step(dt);
            const float stepMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            physicsMs += (stepMs - physicsMs) * 0.05f;

            const auto rv = spot.art->rootVelocity();
            speed += (std::hypot(rv[0], rv[1]) - speed) * std::min(1.f, dt * 4.f);
            downFor = uprightness(*spot.art) < 0.4f ? downFor + dt : 0.f;
            fallen = downFor > 1.f;

            // Follow camera: the orbit target trails the body, and the orbit turns with
            // Spot's smoothed heading so "forward" stays away from the camera. Dragging
            // still orbits freely on top of that.
            Vector3 p;
            base.getWorldPosition(p);
            p.z += 0.2f;
            const Vector3 prev = follow;
            follow.lerp(p, 1.f - std::exp(-8.f * dt));
            const float yaw = yawOf(spot.art->rootState());
            const float dYaw = wrapAngle(yaw - camYaw) * (1.f - std::exp(-1.5f * dt));
            camYaw = wrapAngle(camYaw + dYaw);
            Vector3 offset = camera->position;
            offset.sub(prev);
            offset.applyAxisAngle(Vector3(0, 0, 1), dYaw);
            camera->position.copy(follow).add(offset);
            controls.target.copy(follow);
            controls.update();
            // keep the camera above the ground it looks across
            const float floor = land.heightAt(camera->position.x, camera->position.y) + 0.35f;
            if (camera->position.z < floor) camera->position.z = floor;

            sunTarget->position.copy(p);
            sun->position.copy(p).addScaledVector(sunDir, 40.f);

            scanDots->visible = showScan && ctrl.holdTicksLeft() == 0;// no scan is read during the spawn settle
            if (scanDots->visible) {
                const auto& pts = ctrl.scanPoints();
                for (int i = 0; i < N_SCAN; ++i) {
                    m.makeTranslation(pts[i].x, pts[i].y, pts[i].z + 0.012f);
                    scanDots->setMatrixAt(i, m);
                }
                scanDots->instanceMatrix()->needsUpdate();
            }

            renderer->render(*scene, *camera);
            if (!sunAligned && renderer->envSunFound()) {
                const Vector3 d = renderer->envSunDirection();// unit, toward the sun
                if (d.z > 0.15f) sunDir.copy(d);
                sunAligned = true;
                std::cout << "[spot] sun aligned to the sky: (" << d.x << ", " << d.y << ", " << d.z << ")\n";
            }
            gui.render();
        });
        return 0;
    }

}// namespace

int main(int argc, char** argv) {
    std::string policyPath = defaultPolicyPath();
    std::string urdfPath;
    int checkTicks = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--policy" && i + 1 < argc) policyPath = argv[++i];
        else if (a == "--urdf" && i + 1 < argc) urdfPath = argv[++i];
        else if (a == "--data" && i + 1 < argc) dataDir = argv[++i];
        else if (a == "--check") checkTicks = (i + 1 < argc) ? std::stoi(argv[++i]) : 2500;
    }
    if (urdfPath.empty()) urdfPath = defaultUrdfPath();

    SpotPolicy policy;
    try {
        policy = SpotPolicy::load(policyPath);
    } catch (const std::exception& e) {
        std::cerr << "failed to load policy: " << e.what()
                  << "\n(spot_policy.tpnn ships beside this example; regenerate with "
                     "export_spot_policy.py, or pass --policy <path>)\n";
        return 1;
    }
    std::cout << "[spot] policy " << policyPath << "  in=" << policy.inputDim()
              << " out=" << policy.outputDim() << "\n";
    std::cout << "[spot] robot  " << urdfPath << "\n";

    // The plant and the contract can disagree (a URDF missing a joint the policy
    // drives, a policy exported for a different observation): both throw with the
    // mismatch named rather than driving the wrong leg or reading past an array.
    try {
        return checkTicks > 0 ? runCheck(policy, urdfPath, checkTicks)
                              : runInteractive(policy, urdfPath);
    } catch (const std::exception& e) {
        std::cerr << "spot: " << e.what() << "\n";
        return 1;
    }
}
