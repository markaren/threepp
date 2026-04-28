// PhysX integration showcase: instanced wall, falling spheres, SPACE to throw.
// PxVehicle2 plugs in via world.onPreSubstep without touching PhysxWorld.

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    constexpr int kWallCols = 14;
    constexpr int kWallRows = 8;
    constexpr float kWallBox = 1.0f;

    void layoutWall(InstancedMesh& wall) {
        Matrix4 m;
        const float half = kWallBox * 0.5f;
        const float xOff = -kWallCols * kWallBox * 0.5f + half;
        size_t i = 0;
        for (int row = 0; row < kWallRows; ++row) {
            for (int col = 0; col < kWallCols; ++col) {
                m.setPosition(xOff + col * kWallBox, half + row * kWallBox, 0);
                wall.setMatrixAt(i++, m);
            }
        }
        wall.instanceMatrix()->needsUpdate();
        wall.computeBoundingSphere();
    }

    void resetWallActors(const std::vector<PxRigidActor*>& actors, const InstancedMesh& wall) {
        Matrix4 m;
        Vector3 pos, scale;
        Quaternion rot;
        for (size_t i = 0; i < actors.size(); ++i) {
            wall.getMatrixAt(i, m);
            m.decompose(pos, rot, scale);
            auto* dyn = actors[i]->is<PxRigidDynamic>();
            if (!dyn) continue;
            dyn->setGlobalPose(toPxTransform(pos, rot));
            dyn->setLinearVelocity(PxVec3(0));
            dyn->setAngularVelocity(PxVec3(0));
            dyn->wakeUp();
        }
    }
}// namespace

int main() {

    Canvas canvas("PhysX Demo", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);
    renderer->autoClear = false;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 1000);
    camera->position.set(20, 12, 25);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 4, 0);

    auto sun = DirectionalLight::create(0xffffff, 1.5f);
    sun->position.set(10, 20, 10);
    scene->add(sun);
    scene->add(AmbientLight::create(0xffffff, 0.3f));

    PhysxWorld world;

    auto groundMat = MeshLambertMaterial::create();
    groundMat->color = Color::darkgray;
    auto ground = Mesh::create(BoxGeometry::create(100, 1, 100), groundMat);
    ground->position.y = -0.5f;
    scene->add(ground);
    world.addStatic(*ground);

    auto wallMat = MeshPhongMaterial::create();
    wallMat->color = Color::saddlebrown;
    auto wall = InstancedMesh::create(
            BoxGeometry::create(kWallBox, kWallBox, kWallBox),
            wallMat,
            kWallCols * kWallRows);
    wall->instanceMatrix()->setUsage(DrawUsage::Dynamic);
    layoutWall(*wall);
    scene->add(wall);
    auto wallActors = world.add(*wall, 500.f);

    auto sphereGeom = SphereGeometry::create(0.5f, 16, 12);
    auto sphereMat = MeshPhongMaterial::create();
    sphereMat->color = Color::orange;
    for (int i = 0; i < 10; ++i) {
        auto m = Mesh::create(sphereGeom, sphereMat);
        m->position.set(-4 + i * 0.9f, 18 + i * 0.6f, 6);
        scene->add(m);
        world.add(*m, 500.f);
    }

    auto throwGeom = SphereGeometry::create(0.6f, 16, 12);
    auto throwMat = MeshPhongMaterial::create();
    throwMat->color = Color::red;

    std::vector<std::shared_ptr<Mesh>> thrown;
    bool throwPending = false;

    KeyAdapter spaceKey(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::SPACE) throwPending = true;
    });
    canvas.addKeyListener(spaceKey);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    Vector3 gravity{0, -9.81f, 0};
    bool resetPressed = false;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float width = 260 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - width, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({width, 0}, 0);
        ImGui::Begin("PhysX Demo");
        ImGui::Text("Press SPACE to throw a ball");
        ImGui::Separator();
        if (ImGui::SliderFloat("Gravity Y", &gravity.y, -30.f, 0.f)) world.setGravity(gravity);
        if (ImGui::Button("Reset wall")) resetPressed = true;
        ImGui::Text("Thrown: %zu", thrown.size());
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

        if (throwPending) {
            throwPending = false;
            Vector3 dir;
            dir.subVectors(controls.target, camera->position).normalize();

            auto m = Mesh::create(throwGeom, throwMat);
            m->position.copy(camera->position);
            scene->add(m);
            auto* body = world.add(*m, 1500.f);
            body->setLinearVelocity(toPxVec3(dir * 40.f));
            thrown.push_back(m);
        }

        if (resetPressed) {
            resetPressed = false;
            layoutWall(*wall);
            resetWallActors(wallActors, *wall);
        }

        world.step(dt);

        renderer->clear();
        renderer->render(*scene, *camera);

        ui.render();
    });
}
