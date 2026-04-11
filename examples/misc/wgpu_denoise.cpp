// Hybrid Deferred Showcase — rotating object on a pedestal with multiple
// analytical lights.  Demonstrates noise-free direct lighting in hybrid mode
// during motion vs full path tracing when static.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    // Ground plane — large, slightly rough
    auto makeGround() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.4f, 0.4f, 0.4f)},
                {"roughness", 0.8f},
                {"metalness", 0.0f},
        });
        auto mesh = Mesh::create(PlaneGeometry::create(20.f, 20.f), mat);
        mesh->rotation.x = -math::PI / 2.f;
        return mesh;
    }

    // Pedestal — cylinder
    auto makePedestal() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.7f, 0.7f, 0.72f)},
                {"roughness", 0.3f},
                {"metalness", 0.0f},
        });
        auto mesh = Mesh::create(CylinderGeometry::create(1.0f, 1.2f, 1.5f, 48), mat);
        mesh->position.set(0.f, 0.75f, 0.f);
        return mesh;
    }

    // Hero object — torus knot (interesting shape, shows lighting well)
    auto makeHeroObject() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.9f, 0.15f, 0.1f)},
                {"roughness", 0.9f},
                {"metalness", 0.2f},
        });
        auto mesh = Mesh::create(TorusKnotGeometry::create(0.7f, 0.25f, 128, 32), mat);
        mesh->position.set(0.f, 2.8f, 0.f);
        return mesh;
    }

    // Back wall to catch light and show shadows (thin box so it blocks light from both sides)
    auto makeBackWall() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.75f, 0.75f, 0.75f)},
                {"roughness", 0.9f},
        });
        auto mesh = Mesh::create(BoxGeometry::create(12.f, 8.f, 0.1f), mat);
        mesh->position.set(0.f, 4.f, -5.f);
        return mesh;
    }

    // Glass sphere beside the pedestal
    auto makeGlassSphere() {
        auto mat = MeshPhysicalMaterial::create({
                {"color", Color::steelblue},
                {"transmission", 0.95f},
                {"ior", 1.5f},
                {"roughness", 0.0f},
                {"metalness", 0.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(1.5f, 48, 48), mat);
        mesh->position.set(-4.5f, 1.5f, 1.5f);
        return mesh;
    }

    // Small metal sphere on the other side
    auto makeMetalSphere() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.95f, 0.85f, 0.4f)},
                {"roughness", 0.05f},
                {"metalness", 1.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.4f, 48, 48), mat);
        mesh->position.set(2.2f, 0.4f, 1.8f);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("Hybrid Deferred Showcase",
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", false}});

    WgpuRenderer renderer(canvas);

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(0.0f);
    pathTracer.setExposure(1.0f);
    pathTracer.setDenoiserEnabled(true);
    pathTracer.setMaxBounces(4);
    pathTracer.setReSTIREnabled(true);
    pathTracer.setReSTIRGIEnabled(true);
    // pathTracer.setFireflyClamp(0.001);

    // ---- Scene ----
    Scene scene;
    scene.background = Color(0.02f, 0.02f, 0.05f);

    scene.add(makeGround());
    scene.add(makePedestal());
    scene.add(makeBackWall());
    scene.add(makeGlassSphere());
    scene.add(makeMetalSphere());

    auto hero = makeHeroObject();
    scene.add(hero);

    // ---- Analytical Lights (3 colored lights for clear deferred hybrid demo) ----

    // Warm key light — spotlight from upper right
    Object3D target;
    target.position.set(0.f, 2.f, 0.f);

    auto keyLight = SpotLight::create(Color(1.0f, 0.9f, 0.7f), 2.0f, 15.f, math::PI / 6.f, 0.3f, 1.f);
    keyLight->position.set(4.f, 7.f, 3.f);
    keyLight->setTarget(target);

    scene.add(keyLight);

    // Cool fill light — point light from the left
    auto fillLight = PointLight::create(Color(0.4f, 0.6f, 1.0f), 1.0f);
    fillLight->distance = 12.f;
    fillLight->position.set(-4.f, 4.f, 2.f);
    scene.add(fillLight);

    // Rim/back light — warm point light behind
    auto rimLight = PointLight::create(Color(1.0f, 0.5f, 0.2f), 0.8f);
    rimLight->distance = 10.f;
    rimLight->position.set(1.f, 5.f, -4.f);
    scene.add(rimLight);

    // ---- Camera ----
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 3.5f, 8.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 2.f, 0.f);
    controls.update();

    // ---- UI ----
    bool denoiserOn = pathTracer.denoiserEnabled();
    bool temporalDenoiserOn = pathTracer.temporalDenoiser();
    bool restirOn = pathTracer.restirEnabled();
    bool restirGIOn = pathTracer.restirGiEnabled();
    bool rotating = true;
    float rotSpeed = 0.5f;
    int maxBounces = pathTracer.maxBounces();
    float exposure = pathTracer.exposure();
    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});
        ImGui::Begin("Hybrid Showcase");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frames: %d", pathTracer.frameCount());
        ImGui::Separator();

        if (ImGui::Checkbox("Rotate object", &rotating)) {}
        if (rotating) {
            ImGui::SliderFloat("Speed", &rotSpeed, 0.0f, 3.0f);
        }
        ImGui::Separator();

        if (ImGui::Checkbox("Denoiser", &denoiserOn))
            pathTracer.setDenoiserEnabled(denoiserOn);
        if (ImGui::Checkbox("Temporal Denoiser", &temporalDenoiserOn))
            pathTracer.setTemporalDenoiser(temporalDenoiserOn);
        if (ImGui::Checkbox("ReSTIR DI", &restirOn))
            pathTracer.setReSTIREnabled(restirOn);
        if (ImGui::Checkbox("ReSTIR GI", &restirGIOn))
            pathTracer.setReSTIRGIEnabled(restirGIOn);
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
            pathTracer.setExposure(exposure);
        if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 6))
            pathTracer.setMaxBounces(maxBounces);

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        pathTracer.setSize({ns.width(), ns.height()});
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();

        // Rotate the hero object
        if (rotating) {
            hero->rotation.y += rotSpeed * dt;
        }

        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        controls.update();
        pathTracer.render(scene, camera);
        ui.render();
    });
}
