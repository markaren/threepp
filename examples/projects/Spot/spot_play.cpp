// Play a trained Boston Dynamics Spot policy natively in C++ — no Python, no
// torch — under either the OpenGL or the Vulkan renderer. The policy is loaded
// from a flat .tpnn (exported by export_spot_policy.py) and run by SpotPolicy;
// the scene/contract is SpotScene.hpp. This is the C++ twin of
// python/examples/spot/spot_deploy.py.
//
//   spot_play                 # interactive window, OpenGL
//   spot_play --vulkan        # interactive window, Vulkan (if built with it)
//   spot_play --policy P.tpnn # use a specific exported policy
//   spot_play --check 200     # headless smoke: walk forward, assert upright + moved
//
// DRIVE (body frame, +x fwd / +y left):  arrows move/strafe, N / M turn.

#include "SpotScene.hpp"

#include "threepp/threepp.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/input/KeyListener.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
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

    // Where spot_deploy.py caches the URDF link visuals (link_models/*.obj).
    std::string defaultAssetsDir() {
        const char* home = std::getenv("USERPROFILE");
        if (!home) home = std::getenv("HOME");
        return home ? std::string(home) + "/.cache/threepp/spot" : std::string();
    }

    // Track currently-held keys so the per-frame command can poll them.
    struct HeldKeys: KeyListener {
        std::set<Key> down;
        void onKeyPressed(KeyEvent e) override { down.insert(e.key); }
        void onKeyReleased(KeyEvent e) override { down.erase(e.key); }
        [[nodiscard]] bool is(Key k) const { return down.count(k) > 0; }
    };

    PhysxWorld::Settings spotWorldSettings() {
        PhysxWorld::Settings s;
        s.gravity = Vector3(0, 0, -9.81f);// Z-up, as Isaac / the Python deploy
        s.fixedTimestep = 0.002f;         // 10 substeps per 0.02 s control tick
        s.maxSubSteps = 20;
        return s;
    }

    // Headless: stand, walk forward N ticks, report base pose + uprightness.
    int runCheck(const SpotPolicy& policy, int steps) {
        PhysxWorld world(spotWorldSettings());
        auto ground = Mesh::create(BoxGeometry::create(80, 80, 1.0f), MeshStandardMaterial::create());
        ground->position.set(0, 0, -0.5f);
        world.addStatic(*ground);

        SpotRobot spot = buildSpot(world);
        SpotController ctrl(*spot.art, policy);
        ctrl.hold(world, 150);// stand up
        for (int i = 0; i < steps; ++i) ctrl.step(world, {1.0f, 0.0f, 0.0f});

        const auto rs = spot.art->rootState();
        float R[3][3];
        quatToR(rs[3], rs[4], rs[5], rs[6], R);
        const float upZ = R[2][2];// body local-Z·world-Z; >0.5 => still upright
        const bool upright = upZ > 0.5f;
        const bool standing = rs[2] > 0.35f;
        const bool moved = rs[0] > 0.2f;
        std::cout << "[check] after " << steps << " fwd ticks: base=("
                  << rs[0] << "," << rs[1] << "," << rs[2] << ")  upZ=" << upZ
                  << "  upright=" << upright << " standing=" << standing << " moved=" << moved << "\n";
        const bool ok = upright && standing && moved;
        std::cout << (ok ? "SPOT C++ PLAY CHECK: OK\n" : "SPOT C++ PLAY CHECK: FAIL\n");
        return ok ? 0 : 1;
    }

    int runInteractive(const SpotPolicy& policy, const std::string& assetsDir) {

        PhysxWorld world(spotWorldSettings());
        auto ground = Mesh::create(BoxGeometry::create(80, 80, 1.0f), MeshStandardMaterial::create());
        ground->position.set(0, 0, -0.5f);
        world.addStatic(*ground);

        SpotRobot spot = buildSpot(world, 0.f, 0.f, assetsDir);// assetsDir -> render the URDF visuals
        SpotController ctrl(*spot.art, policy);
        ctrl.hold(world, 150);

        Canvas canvas(Canvas::Parameters().title("threepp - Spot (native C++ policy)").size(1100, 640).antialiasing(4));
        auto renderer = createRenderer(canvas);
        renderer->shadowMap().enabled = true;
        renderer->toneMapping = ToneMapping::ACESFilmic;
        renderer->toneMappingExposure = 1.1f;

        auto scene = Scene::create();
        scene->fog = FogExp2(0x7ea9d0, .05f);
        scene->background = Color(0x7ea9d0);
        scene->add(HemisphereLight::create(0xd0e4f7, 0x4a5a6a, 1.15f));
        auto sun = DirectionalLight::create(0xffffff, 2.8f);
        sun->position.set(6, -5, 11);
        sun->castShadow = true;
        // The default directional shadow camera is a 10×10 box at the origin, so the
        // shadow vanishes once Spot walks away. Keep a tight frustum and move the light
        // + its target with Spot each frame (below) for a crisp, always-centred shadow.
        if (auto* sc = dynamic_cast<OrthographicCamera*>(sun->shadow->camera.get())) {
            sc->left = -4.f;
            sc->right = 4.f;
            sc->top = 4.f;
            sc->bottom = -4.f;
            sc->updateProjectionMatrix();
        }
        sun->shadow->mapSize.set(2048, 2048);
        scene->add(sun);
        auto sunTarget = Group::create();// the light aims here; we park it under Spot
        scene->add(sunTarget);
        sun->setTarget(*sunTarget);
        const Vector3 sunOffset(6, -5, 11);// keeps the light DIRECTION constant while following

        auto grid = GridHelper::create(120, 120, 0x3a4654, 0x2c343d);
        grid->rotation.x = math::PI/2;
        scene->add(grid);

        auto floorMat = MeshStandardMaterial::create();
        floorMat->color = Color(0x4a6f9a);
        floorMat->roughness = 0.9f;
        floorMat->metalness = 0.0f;
        auto floor = Mesh::create(PlaneGeometry::create(120, 120), floorMat);// XY plane (Z-up ground)
        floor->receiveShadow = true;
        scene->add(floor);

        for (auto& m : spot.meshes) scene->add(m);// castShadow handled in buildSpot (primitive vs OBJ)

        auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.01f, 300.f);
        camera->up.set(0, 0, 1);// Z-up world
        camera->position.set(-3.0f, 0.0f, 1.4f);
        camera->lookAt(Vector3(0, 0, 0.3f));

        canvas.onWindowResize([&](WindowSize size) {
            camera->aspect = size.aspect();
            camera->updateProjectionMatrix();
            renderer->setSize(size);
        });

        HeldKeys keys;
        canvas.addKeyListener(keys);

        constexpr float BACK = 2.8f, HEIGHT = 1.5f, LAG = 0.08f;
        bool hasLock = false;
        float headingLock = 0.f;

        canvas.animate([&] {
            // velocity command [vx, vy, wz] in Spot's body frame (+x fwd, +y left)
            const float vx = (keys.is(Key::UP) ? 1.5f : 0.f) - (keys.is(Key::DOWN) ? 1.0f : 0.f);
            const float vy = (keys.is(Key::LEFT) ? 1.0f : 0.f) - (keys.is(Key::RIGHT) ? 1.0f : 0.f);
            const float turn = (keys.is(Key::N) ? 1.5f : 0.f) - (keys.is(Key::M) ? 1.5f : 0.f);

            const auto rs = spot.art->rootState();
            float R[3][3];
            quatToR(rs[3], rs[4], rs[5], rs[6], R);
            const float yaw = std::atan2(R[1][0], R[0][0]);
            // Hold heading when not actively turning (the policy only regulates
            // yaw RATE to 0, so any bias slowly spirals); a light P keeps it straight.
            float wz;
            if (turn != 0.f) {
                wz = turn;
                headingLock = yaw;
                hasLock = true;
            } else {
                if (!hasLock) {
                    headingLock = yaw;
                    hasLock = true;
                }
                float err = std::fmod(yaw - headingLock + math::PI, 2.f * math::PI);
                if (err < 0) err += 2.f * math::PI;
                err -= math::PI;
                wz = std::clamp(-2.0f * err, -1.0f, 1.0f);
            }
            ctrl.step(world, {vx, vy, wz});

            // keep the sun's shadow box centred under Spot (constant light direction)
            Vector3 p(rs[0], rs[1], rs[2]);
            sunTarget->position.set(p.x, p.y, 0.f);
            sun->position.set(p.x + sunOffset.x, p.y + sunOffset.y, sunOffset.z);

            // chase cam: trail BACK m behind Spot's heading at HEIGHT, look at the body
            Vector3 fwd(R[0][0], R[1][0], 0.f);// body +x in world, levelled
            if (fwd.length() > 1e-6f) fwd.normalize();
            else fwd.set(1, 0, 0);
            Vector3 desired(p.x - fwd.x * BACK, p.y - fwd.y * BACK, p.z + HEIGHT);
            camera->position.lerp(desired, LAG);
            camera->lookAt(Vector3(p.x + fwd.x * 0.4f, p.y + fwd.y * 0.4f, p.z + 0.15f));

            renderer->render(*scene, *camera);
        });
        return 0;
    }

}// namespace

int main(int argc, char** argv) {
    std::string policyPath = defaultPolicyPath();
    std::string assetsDir = defaultAssetsDir();
    int checkSteps = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--policy" && i + 1 < argc) policyPath = argv[++i];
        else if (a == "--assets" && i + 1 < argc) assetsDir = argv[++i];
        else if (a == "--no-visuals") assetsDir.clear();
        else if (a == "--check") checkSteps = (i + 1 < argc) ? std::stoi(argv[++i]) : 200;
    }

    SpotPolicy policy;
    try {
        policy = SpotPolicy::load(policyPath);
    } catch (const std::exception& e) {
        std::cerr << "failed to load policy: " << e.what()
                  << "\n(run export_spot_policy.py first, or pass --policy <path>)\n";
        return 1;
    }
    std::cout << "[spot] policy " << policyPath << "  in=" << policy.inputDim()
              << " out=" << policy.outputDim() << "\n";

    return checkSteps > 0 ? runCheck(policy, checkSteps) : runInteractive(policy, assetsDir);
}
