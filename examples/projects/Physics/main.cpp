// PhysX Playground — an interactive sandbox for the threepp <-> PhysX integration.
//
//   • a crate pyramid to knock down
//   • a wrecking ball on a cable (PxDistanceJoint) — shoot or shove it to swing
//   • a constantly-sweeping kinematic paddle (setKinematicTarget each substep)
//   • a domino run
//   • a bin of bouncy props (custom restitution PxMaterial)
//
// Interaction:
//   • LEFT-DRAG an object to grab and fling it (orbit still works on empty space)
//   • SPACE  — shoot a bouncy ball along the cursor ray
//   • G      — rain a burst of mixed shapes
//   • R      — reset
// PxVehicle2 plugs into the same world via world.onPreSubstep — see physx_vehicle.

#include "threepp/threepp.hpp"

#include "threepp/core/Raycaster.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    // Deterministic RNG so the scene lays out the same every run.
    std::mt19937 rng(1337);
    float frand(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }
    int irand(int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); }

    const std::array<Color, 6> kCrateColors{
            Color(0xC8783C), Color(0xD9A05B), Color(0xA85B32),
            Color(0xE0B080), Color(0xB5732E), Color(0x8C5A2B)};
    const std::array<Color, 6> kBrightColors{
            Color(0xE54B4B), Color(0x3CA0E5), Color(0x49C66A),
            Color(0xE5C04B), Color(0xA64BE5), Color(0xE5814B)};

    std::shared_ptr<MeshStandardMaterial> matte(const Color& c, float rough = 0.75f) {
        auto m = MeshStandardMaterial::create();
        m->color = c;
        m->roughness = rough;
        m->metalness = 0.f;
        return m;
    }

    std::shared_ptr<Mesh> shadowMesh(const std::shared_ptr<BufferGeometry>& g,
                                     const std::shared_ptr<Material>& m) {
        auto mesh = Mesh::create(g, m);
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        return mesh;
    }

    struct Body {
        std::shared_ptr<Mesh> mesh;
        PxRigidDynamic* actor;
    };

}// namespace

int main() {

    Canvas canvas("PhysX Playground", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color(0x9fb8d4);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);
    camera->position.set(22, 14, 26);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 3, 0);

    // Sun with a shadow frustum covering the play area, plus a soft fill.
    constexpr float kArea = 30.f;
    auto sun = DirectionalLight::create(0xffffff, 2.2f);
    sun->position.set(18, 30, 14);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    sun->shadow->bias = -0.0004f;
    sun->shadow->camera->as<OrthographicCamera>()->left = -kArea;
    sun->shadow->camera->as<OrthographicCamera>()->right = kArea;
    sun->shadow->camera->as<OrthographicCamera>()->top = kArea;
    sun->shadow->camera->as<OrthographicCamera>()->bottom = -kArea;
    sun->shadow->camera->nearPlane = 1.f;
    sun->shadow->camera->farPlane = 120.f;
    sun->shadow->camera->updateProjectionMatrix();
    scene->add(sun);
    scene->add(AmbientLight::create(0xb9cbe0, 0.6f));

    PhysxWorld world;

    // Extra contact materials (staticFric, dynamicFric, restitution).
    PxMaterial* bouncyMat = world.physics().createMaterial(0.4f, 0.3f, 0.85f);

    // Ground (visual + static collider) with a faint grid for depth reference.
    auto ground = shadowMesh(BoxGeometry::create(120, 1, 120), matte(Color(0x6b7a55)));
    ground->castShadow = false;
    ground->position.y = -0.5f;
    scene->add(ground);
    world.addStatic(*ground);
    auto grid = GridHelper::create(120, 60, Color::gray, Color(0x808080));
    grid->position.y = 0.01f;
    scene->add(grid);

    // ---- spawn helpers ----------------------------------------------------

    std::vector<Body> structure;            // persistent: pyramid + dominoes (reset in place)
    std::vector<PxTransform> structureInit; // their start poses
    std::vector<Body> props;                // transient: shot / rained / scattered (cleared on reset)

    auto addStructure = [&](const std::shared_ptr<Mesh>& mesh, float density) {
        scene->add(mesh);
        auto* actor = world.add(*mesh, density);
        structure.push_back({mesh, actor});
        structureInit.push_back(actor->getGlobalPose());
    };
    auto addProp = [&](const std::shared_ptr<Mesh>& mesh, float density,
                       PxMaterial* mat = nullptr) -> PxRigidDynamic* {
        scene->add(mesh);
        auto* actor = world.add(*mesh, density, mat);
        props.push_back({mesh, actor});
        return actor;
    };

    // Crate pyramid centred at the origin (base 6 wide, 1 m cubes).
    auto buildPyramid = [&] {
        constexpr int base = 6;
        constexpr float s = 1.f;
        for (int row = 0; row < base; ++row) {
            const int n = base - row;
            const float x0 = -(n - 1) * 0.5f * s;
            for (int col = 0; col < n; ++col) {
                auto box = shadowMesh(BoxGeometry::create(s * 0.98f, s * 0.98f, s * 0.98f),
                                      matte(kCrateColors[(row + col) % kCrateColors.size()]));
                box->position.set(x0 + col * s, s * 0.5f + row * s, -2.f);
                addStructure(box, 120.f);
            }
        }
    };

    // Domino run along -X.
    auto buildDominoes = [&] {
        constexpr int n = 14;
        for (int i = 0; i < n; ++i) {
            auto d = shadowMesh(BoxGeometry::create(0.9f, 1.8f, 0.22f),
                                matte(Color(0x2f7fb5)));
            d->position.set(-13.f + i * 0.7f, 0.9f, 6.f);
            addStructure(d, 50.f);
        }
    };

    buildPyramid();
    buildDominoes();

    // Wrecking ball: heavy sphere held by a distance joint to a static anchor up
    // high, so it swings freely on a cable. A Line tracks the cable each frame.
    const PxVec3 ballAnchor(8.f, 13.f, -2.f);
    constexpr float kCable = 8.f;
    auto ballMesh = shadowMesh(SphereGeometry::create(1.2f, 24, 18), [] {
        auto m = MeshStandardMaterial::create();
        m->color = Color(0x303338);
        m->roughness = 0.35f;
        m->metalness = 0.9f;
        return m;
    }());
    ballMesh->position.set(ballAnchor.x, ballAnchor.y - kCable, ballAnchor.z);
    scene->add(ballMesh);
    auto* ballActor = world.add(*ballMesh, 1000.f);// ~7 t — smashes crates
    ballActor->setLinearDamping(0.15f);
    ballActor->setAngularDamping(0.2f);
    const PxTransform ballInit = ballActor->getGlobalPose();
    {
        auto* anchorActor = world.physics().createRigidStatic(PxTransform(ballAnchor));
        world.scene().addActor(*anchorActor);
        auto* j = PxDistanceJointCreate(world.physics(),
                                        anchorActor, PxTransform(PxIdentity),
                                        ballActor, PxTransform(PxIdentity));
        j->setMaxDistance(kCable);
        j->setDistanceJointFlag(PxDistanceJointFlag::eMAX_DISTANCE_ENABLED, true);
        j->setDistanceJointFlag(PxDistanceJointFlag::eMIN_DISTANCE_ENABLED, false);
    }
    // Visual gantry post + cable.
    auto post = shadowMesh(BoxGeometry::create(0.4f, ballAnchor.y, 0.4f), matte(Color(0x44484f)));
    post->position.set(ballAnchor.x + 1.2f, ballAnchor.y * 0.5f, ballAnchor.z);
    scene->add(post);
    auto arm = shadowMesh(BoxGeometry::create(1.6f, 0.4f, 0.4f), matte(Color(0x44484f)));
    arm->position.set(ballAnchor.x + 0.4f, ballAnchor.y, ballAnchor.z);
    scene->add(arm);
    auto cableGeom = BufferGeometry::create();
    cableGeom->setAttribute("position", FloatBufferAttribute::create({ballAnchor.x, ballAnchor.y, ballAnchor.z,
                                                                       ballInit.p.x, ballInit.p.y, ballInit.p.z},
                                                                      3));
    cableGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    auto cableMat = LineBasicMaterial::create();
    cableMat->color = Color(0x101010);
    auto cable = Line::create(cableGeom, cableMat);
    cable->frustumCulled = false;// endpoint moves with the ball; skip stale-bounds cull
    scene->add(cable);

    // Spinning kinematic paddle — a lazy-susan bar that keeps the scene alive.
    float paddleAngle = 0.f, paddleSpeed = 1.6f;
    bool paddleOn = true;
    const PxVec3 paddlePos(12.f, 0.7f, 11.f);
    auto paddleMesh = shadowMesh(BoxGeometry::create(7.f, 0.5f, 0.5f), matte(Color(0xd24b4b), 0.4f));
    paddleMesh->position.set(paddlePos.x, paddlePos.y, paddlePos.z);
    scene->add(paddleMesh);
    auto* paddleActor = world.physics().createRigidDynamic(PxTransform(paddlePos));
    paddleActor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
    {
        auto* ps = world.physics().createShape(PxBoxGeometry(3.5f, 0.25f, 0.25f), world.defaultMaterial());
        paddleActor->attachShape(*ps);
        ps->release();
    }
    world.scene().addActor(*paddleActor);
    world.bind(*paddleMesh, *paddleActor);
    world.onPreSubstep([&](float dt) {
        if (paddleOn) paddleAngle += paddleSpeed * dt;
        paddleActor->setKinematicTarget(PxTransform(paddlePos, PxQuat(paddleAngle, PxVec3(0, 1, 0))));
    });

    // A scatter of bouncy props near the paddle for it to swat around.
    auto spawnRandomProp = [&](const Vector3& pos) {
        const Color c = kBrightColors[irand(0, static_cast<int>(kBrightColors.size()) - 1)];
        std::shared_ptr<Mesh> mesh;
        switch (irand(0, 2)) {
            case 0: mesh = shadowMesh(BoxGeometry::create(0.7f, 0.7f, 0.7f), matte(c, 0.5f)); break;
            case 1: mesh = shadowMesh(SphereGeometry::create(0.45f, 18, 14), matte(c, 0.4f)); break;
            default: mesh = shadowMesh(CapsuleGeometry::create(0.35f, 0.7f), matte(c, 0.5f)); break;
        }
        mesh->position.copy(pos);
        return addProp(mesh, 70.f, bouncyMat);
    };
    for (int i = 0; i < 9; ++i)
        spawnRandomProp({frand(8.f, 16.f), frand(2.f, 5.f), frand(8.f, 15.f)});

    // ---- interaction state ------------------------------------------------

    Raycaster raycaster;
    Vector2 mouseNdc{2.f, 2.f};// off-screen until moved
    bool tryGrab = false, releaseGrab = false, shootPending = false, rainPending = false, resetPending = false;

    PxRigidDynamic* grabbed = nullptr;
    float grabDepth = 0.f;
    Vector3 grabTargetPrev, grabVel;

    struct PlaygroundMouse : MouseListener {
        Canvas& canvas;
        Vector2& ndc;
        bool &down, &up;
        PlaygroundMouse(Canvas& c, Vector2& n, bool& d, bool& u) : canvas(c), ndc(n), down(d), up(u) {}
        void setNdc(const Vector2& pos) {
            const auto s = canvas.size();
            ndc.x = pos.x / static_cast<float>(s.width()) * 2.f - 1.f;
            ndc.y = -(pos.y / static_cast<float>(s.height())) * 2.f + 1.f;
        }
        void onMouseMove(const Vector2& pos) override { setNdc(pos); }
        void onMouseDown(int button, const Vector2& pos) override {
            if (button == 0 && !ImGui::GetIO().WantCaptureMouse) {
                setNdc(pos);
                down = true;
            }
        }
        void onMouseUp(int button, const Vector2&) override {
            if (button == 0) up = true;
        }
    } mouseListener(canvas, mouseNdc, tryGrab, releaseGrab);
    canvas.addMouseListener(mouseListener);

    KeyAdapter keys(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::SPACE) shootPending = true;
        if (evt.key == Key::G) rainPending = true;
        if (evt.key == Key::R) resetPending = true;
    });
    canvas.addKeyListener(keys);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    auto clearProps = [&] {
        for (auto& b : props) {
            if (b.actor == grabbed) grabbed = nullptr;
            b.actor->getScene()->removeActor(*b.actor);
            world.unbind(*b.mesh);
            scene->remove(*b.mesh);
            b.actor->release();
        }
        props.clear();
    };

    Vector3 gravity{0, -9.81f, 0};

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float w = 270 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - w, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({w, 0}, 0);
        ImGui::Begin("PhysX Playground");
        ImGui::TextWrapped("Left-drag: grab & fling");
        ImGui::Text("SPACE: shoot   G: rain   R: reset");
        ImGui::Separator();
        if (ImGui::SliderFloat("Gravity Y", &gravity.y, -30.f, 0.f)) world.setGravity(gravity);
        ImGui::Checkbox("Paddle", &paddleOn);
        ImGui::SameLine();
        ImGui::SliderFloat("##spd", &paddleSpeed, 0.f, 5.f, "%.1f rad/s");
        if (ImGui::Button("Rain burst (G)")) rainPending = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset (R)")) resetPending = true;
        ImGui::Text("Loose props: %zu", props.size());
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        // Begin a grab: PhysX raycast along the cursor ray; grab the first loose
        // dynamic (anything but the ground and the kinematic paddle — the wrecking
        // ball included). Suspend orbit by toggling enableRotate, NOT enabled:
        // OrbitControls' mouse-up listener only tears down its gesture while
        // enabled is true, so flipping enabled mid-drag strands it in ROTATE.
        if (tryGrab) {
            tryGrab = false;
            raycaster.setFromCamera(mouseNdc, *camera);
            PxRaycastBuffer hit;
            const Vector3 dir = Vector3(raycaster.ray.direction).normalize();
            if (world.scene().raycast(toPxVec3(raycaster.ray.origin), toPxVec3(dir), 300.f, hit) && hit.hasBlock) {
                auto* dyn = hit.block.actor->is<PxRigidDynamic>();
                const bool kinematic = dyn && dyn->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
                if (dyn && !kinematic) {
                    grabbed = dyn;
                    grabDepth = hit.block.distance;
                    grabbed->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
                    grabTargetPrev = raycaster.ray.origin + dir * grabDepth;
                    grabVel.set(0, 0, 0);
                    controls.enableRotate = false;
                }
            }
        }

        // Drive the grabbed body to the cursor at the grab depth, tracking velocity.
        if (grabbed && !releaseGrab) {
            raycaster.setFromCamera(mouseNdc, *camera);
            const Vector3 dir = Vector3(raycaster.ray.direction).normalize();
            Vector3 target = raycaster.ray.origin + dir * grabDepth;
            // Keep the wrecking ball on its cable sphere so it swings (not snaps) on release.
            if (grabbed == ballActor) {
                const Vector3 anchorV = fromPxVec3(ballAnchor);
                Vector3 fromAnchor = target - anchorV;
                if (fromAnchor.length() > kCable) target = anchorV + fromAnchor.normalize() * kCable;
            }
            grabbed->setKinematicTarget(PxTransform(toPxVec3(target), grabbed->getGlobalPose().q));
            if (dt > 1e-5f) grabVel = (target - grabTargetPrev) * (1.f / dt);
            grabTargetPrev = target;
        }

        // Release: hand momentum back to the body (clamped so it can't explode).
        if (releaseGrab) {
            releaseGrab = false;
            if (grabbed) {
                grabbed->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false);
                if (grabVel.length() > 22.f) grabVel.normalize().multiplyScalar(22.f);
                grabbed->setLinearVelocity(toPxVec3(grabVel));
                grabbed->wakeUp();
                grabbed = nullptr;
            }
            controls.enableRotate = true;
        }

        if (shootPending) {
            shootPending = false;
            raycaster.setFromCamera(mouseNdc, *camera);
            const Vector3 dir = mouseNdc.x <= 1.f
                                        ? Vector3(raycaster.ray.direction).normalize()
                                        : Vector3(controls.target).sub(camera->position).normalize();
            auto ball = shadowMesh(SphereGeometry::create(0.5f, 20, 14),
                                   matte(kBrightColors[irand(0, static_cast<int>(kBrightColors.size()) - 1)], 0.4f));
            ball->position.copy(camera->position + dir * 1.5f);
            auto* a = addProp(ball, 90.f, bouncyMat);
            a->setLinearVelocity(toPxVec3(dir * 42.f));
        }

        if (rainPending) {
            rainPending = false;
            for (int i = 0; i < 12; ++i)
                spawnRandomProp({frand(-8.f, 8.f), frand(14.f, 22.f), frand(-6.f, 8.f)});
        }

        if (resetPending) {
            resetPending = false;
            clearProps();
            for (size_t i = 0; i < structure.size(); ++i) {
                structure[i].actor->setGlobalPose(structureInit[i]);
                structure[i].actor->setLinearVelocity(PxVec3(0));
                structure[i].actor->setAngularVelocity(PxVec3(0));
                structure[i].actor->wakeUp();
            }
            ballActor->setGlobalPose(ballInit);
            ballActor->setLinearVelocity(PxVec3(0));
            ballActor->setAngularVelocity(PxVec3(0));
            ballActor->wakeUp();
            controls.enableRotate = true;// in case reset happened mid-grab
        }

        world.step(dt);

        // Update the cable line to follow the swinging ball.
        {
            const PxVec3 bp = ballActor->getGlobalPose().p;
            auto* posAttr = cableGeom->getAttribute<float>("position");
            posAttr->setXYZ(1, bp.x, bp.y, bp.z);
            posAttr->needsUpdate();
        }

        renderer->render(*scene, *camera);
        ui.render();
    });
}
