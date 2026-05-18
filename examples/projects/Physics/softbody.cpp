// Soft body (PxDeformableVolume) demo. Requires a CUDA-capable GPU — soft
// bodies run on PhysX's GPU dynamics solver. SPACE drops fresh soft bodies.

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxDebugRenderer.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

#include <random>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    // Drop a random soft shape (sphere/torus/cone) at the given world position.
    SoftBody* spawnSoftBody(PhysxWorld& world, Scene& scene,
                            const std::shared_ptr<Material>& mat,
                            const Vector3& spawn,
                            PxDeformableVolumeMaterial* sbMat,
                            std::mt19937& rng) {

        std::uniform_int_distribution<int> kind(0, 2);
        const int k = kind(rng);

        std::shared_ptr<BufferGeometry> geom;
        if (k == 0) {
            geom = SphereGeometry::create(0.8f, 16, 12);
        } else if (k == 1) {
            auto t = TorusGeometry::create(0.7f, 0.3f, 12, 24);
            geom = t;
        } else {
            geom = ConeGeometry::create(0.7f, 1.4f, 24);
        }

        auto mesh = Mesh::create(geom, mat);
        mesh->position.copy(spawn);
        scene.add(mesh);
        return world.addSoftBody(*mesh, sbMat, 10);
    }
}// namespace

int main() {

    Canvas canvas("PhysX Soft Body Demo", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);
    renderer->autoClear = false;

    auto scene = Scene::create();
    scene->background = Color::aliceblue;

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01f, 1000);
    camera->position.set(8, 6, 12);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 1, 0);

    auto sun = DirectionalLight::create(0xffffff, 1.5f);
    sun->position.set(10, 20, 10);
    scene->add(sun);
    // scene->add(AmbientLight::create(0xffffff, 0.4f));

    // GPU dynamics is required for soft bodies. The constructor will throw if
    // there's no CUDA-capable GPU / driver.
    PhysxWorld::Settings settings;
    settings.enableGpuDynamics = true;
    PhysxWorld world(settings);

    PhysxDebugRenderer debugRenderer(world);
    scene->add(debugRenderer);

    auto groundMat = MeshLambertMaterial::create();
    groundMat->color = Color::darkgray;
    auto ground = Mesh::create(BoxGeometry::create(20, 0.5f, 20), groundMat);
    ground->position.y = -0.25f;
    scene->add(ground);
    world.addStatic(*ground);

    // Rigid pillars to bounce off.
    auto pillarMat = MeshPhongMaterial::create();
    pillarMat->color = Color::saddlebrown;
    for (int i = 0; i < 4; ++i) {
        const float a = float(i) * math::TWO_PI / 4.f;
        auto p = Mesh::create(BoxGeometry::create(0.8f, 2.0f, 0.8f), pillarMat);
        p->position.set(std::cos(a) * 3.5f, 1.0f, std::sin(a) * 3.5f);
        scene->add(p);
        world.addStatic(*p);
    }

    auto sbMaterial = MeshPhongMaterial::create();
    sbMaterial->color = Color::orange;
    sbMaterial->flatShading = true;

    auto* sbPhysicsMat = world.createSoftBodyMaterial(/*young=*/ 1.0e6f, /*poisson=*/ 0.45f, /*friction=*/ 0.5f);
    sbPhysicsMat->setDamping(1.0f);
    sbPhysicsMat->setElasticityDamping(0.025f);

    std::mt19937 rng{12345};

    // Initial set so the scene isn't empty on launch.
    for (int i = 0; i < 4; ++i) {
        spawnSoftBody(world, *scene, sbMaterial, {-2.f + i * 1.4f, 5.f + i * 0.6f, 0}, sbPhysicsMat, rng);
    }

    bool spawnPending = false;
    KeyAdapter spaceKey(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::SPACE) spawnPending = true;
    });
    canvas.addKeyListener(spaceKey);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    bool showDebug = false;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float width = 280 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - width, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({width, 0}, 0);
        ImGui::Begin("Soft Body Demo");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();
        ImGui::Text("Press SPACE to drop a soft body");
        ImGui::Separator();
        if (ImGui::Checkbox("Show PhysX debug", &showDebug)) {
            debugRenderer.visible = showDebug;
            if (showDebug) debugRenderer.enableDefaults();
        }
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;

    canvas.animate([&] {
        const float realDt = clock.getDelta();
        fpsAccum += realDt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (spawnPending) {
            spawnPending = false;
            Vector3 dir;
            dir.subVectors(controls.target, camera->position).normalize();
            Vector3 spawn;
            spawn.copy(camera->position).addScaledVector(dir, 5.f);
            spawn.y = std::max(spawn.y, 5.f);
            spawnSoftBody(world, *scene, sbMaterial, spawn, sbPhysicsMat, rng);
        }

        world.step(realDt);
        debugRenderer.update();

        renderer->clear();
        renderer->render(*scene, *camera);
        ui.render();
    });
}
