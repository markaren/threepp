// ReSTIR DI test scene — multiple emissive lights, occluders, varied materials.
// Designed to stress-test reservoir-based direct illumination sampling.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto makePlane(float w, float h, const Color& color, float roughness = 0.9f) {
        auto mat = MeshStandardMaterial::create({{"color", color}, {"roughness", roughness}});
        return Mesh::create(PlaneGeometry::create(w, h), mat);
    }

    // Room: floor + back wall + two side walls + ceiling
    auto makeRoom(float size = 20.f) {
        auto group = Group::create();

        auto floor = makePlane(size, size, Color(0.7f, 0.7f, 0.7f));
        floor->rotation.x = -math::PI / 2.f;
        group->add(floor);

        auto ceiling = makePlane(size, size, Color(0.8f, 0.8f, 0.8f));
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = size;
        group->add(ceiling);

        auto back = makePlane(size, size, Color(0.7f, 0.7f, 0.7f));
        back->position.set(0.f, size / 2.f, -size / 2.f);
        group->add(back);

        auto left = makePlane(size, size, Color(0.6f, 0.1f, 0.1f), 0.95f);
        left->rotation.y = math::PI / 2.f;
        left->position.set(-size / 2.f, size / 2.f, 0.f);
        group->add(left);

        auto right = makePlane(size, size, Color(0.1f, 0.1f, 0.6f), 0.95f);
        right->rotation.y = -math::PI / 2.f;
        right->position.set(size / 2.f, size / 2.f, 0.f);
        group->add(right);

        return group;
    }

    // Emissive sphere light
    auto makeEmissiveLight(const Vector3& pos, const Color& color, float intensity, float radius = 0.4f) {
        auto mat = MeshStandardMaterial::create({
                {"color", color},
                {"emissive", color},
                {"emissiveIntensity", intensity},
                {"roughness", 1.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(radius, 32, 32), mat);
        mesh->position.copy(pos);
        return mesh;
    }

    // Occluder box
    auto makeOccluder(const Vector3& pos, const Vector3& size, const Color& color) {
        auto mat = MeshStandardMaterial::create({{"color", color}, {"roughness", 0.8f}});
        auto mesh = Mesh::create(BoxGeometry::create(size.x, size.y, size.z), mat);
        mesh->position.copy(pos);
        return mesh;
    }

    // Metal sphere
    auto makeMetalSphere(const Vector3& pos, const Color& color, float roughness = 0.05f) {
        auto mat = MeshStandardMaterial::create({
                {"color", color},
                {"roughness", roughness},
                {"metalness", 1.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.8f, 48, 48), mat);
        mesh->position.copy(pos);
        return mesh;
    }

    // Glass sphere
    auto makeGlassSphere(const Vector3& pos) {
        auto mat = MeshPhysicalMaterial::create({
                {"color", Color::white},
                {"transmission", 0.95f},
                {"ior", 1.5f},
                {"roughness", 0.0f},
                {"metalness", 0.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.7f, 48, 48), mat);
        mesh->position.copy(pos);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("ReSTIR DI Test",
                  {{"graphicsApi", GraphicsAPI::WebGPU}, {"vsync", false}});

    WgpuRenderer renderer(canvas);

    WgpuPathTracer pathTracer(renderer, canvas.size());
    pathTracer.setEnvIntensity(1.0f);
    pathTracer.setExposure(1.0f);
    pathTracer.setMaxBounces(6);
    pathTracer.setMode(WgpuPathTracer::Mode::PathTracer);
    pathTracer.setDenoiserEnabled(false);
    pathTracer.setReSTIREnabled(false);
    pathTracer.setFoveatedRendering(false);

    // ---- Scene ----
    Scene scene;
    scene.background = Color::black;
    scene.add(makeRoom(20.f));

    // Multiple emissive lights at different positions and colors
    scene.add(makeEmissiveLight({-4.f, 8.f, -4.f}, Color(1.f, 0.9f, 0.7f), 6.f, 0.5f));  // warm white
    scene.add(makeEmissiveLight({4.f, 6.f, -3.f}, Color(0.3f, 0.5f, 1.f), 8.f, 0.4f));    // blue
    scene.add(makeEmissiveLight({0.f, 3.f, 2.f}, Color(1.f, 0.3f, 0.1f), 5.f, 0.35f));     // red-orange
    scene.add(makeEmissiveLight({-3.f, 2.f, 4.f}, Color(0.2f, 1.f, 0.3f), 4.f, 0.3f));     // green
    scene.add(makeEmissiveLight({5.f, 9.f, 0.f}, Color(1.f, 1.f, 1.f), 3.f, 0.6f));        // white

    // Occluders to cast shadows
    scene.add(makeOccluder({-2.f, 2.f, -2.f}, {2.f, 4.f, 1.f}, Color(0.6f, 0.6f, 0.6f)));
    scene.add(makeOccluder({3.f, 1.5f, 1.f}, {1.5f, 3.f, 1.5f}, Color(0.5f, 0.5f, 0.55f)));
    scene.add(makeOccluder({0.f, 1.f, -4.f}, {3.f, 2.f, 0.5f}, Color(0.65f, 0.6f, 0.55f)));

    // Different material spheres to test BRDF evaluation
    scene.add(makeMetalSphere({-4.f, 0.8f, 2.f}, Color(1.f, 0.84f, 0.f), 0.02f));   // gold mirror
    scene.add(makeMetalSphere({-2.f, 0.8f, 3.f}, Color(0.9f, 0.9f, 0.9f), 0.3f));    // rough silver
    scene.add(makeGlassSphere({1.f, 0.7f, 3.f}));
    scene.add(makeMetalSphere({3.f, 0.8f, 3.f}, Color(0.8f, 0.5f, 0.3f), 0.1f));     // copper

    // Point light for comparison in raytracer mode
    auto ptLight = PointLight::create(Color::white, 0.3f);
    ptLight->position.set(0.f, 18.f, 0.f);
    scene.add(ptLight);

    // ---- Camera ----
    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 6.f, 16.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 4.f, 0.f);
    controls.update();

    // ---- UI ----
    bool restirOn = pathTracer.restirEnabled();
    bool denoiserOn = pathTracer.denoiserEnabled();
    int maxBounces = pathTracer.maxBounces();
    float exposure = pathTracer.exposure();
    float fps = 0.f;
    float fpsAccum = 0.f;
    int fpsFrames = 0;
    int renderMode = 1;// path tracer
    std::vector<std::string> modeNames = {"Raytracer", "PathTracer", "Raster"};
    bool raster = false;

    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            renderMode = (renderMode + 1) % 3;
            raster = (renderMode == 2);
            if (!raster) {
                pathTracer.setMode(renderMode == 1 ? WgpuPathTracer::Mode::PathTracer
                                                    : WgpuPathTracer::Mode::Raytracer);
            }
        }
        if (ev.key == Key::R) {
            restirOn = !restirOn;
            pathTracer.setReSTIREnabled(restirOn);
        }
    });
    canvas.addKeyListener(keyAdapter);

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({});
        ImGui::SetNextWindowSize({});
        ImGui::Begin(modeNames[renderMode].c_str());
        ImGui::Text("FPS: %.1f", fps);
        if (!raster) ImGui::Text("Frames: %d", pathTracer.frameCount());
        ImGui::Separator();

        if (!raster) {
            if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
                pathTracer.setExposure(exposure);
        }

        if (renderMode == 1 && ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("ReSTIR DI (R)", &restirOn))
                pathTracer.setReSTIREnabled(restirOn);
            if (ImGui::Checkbox("Denoiser", &denoiserOn))
                pathTracer.setDenoiserEnabled(denoiserOn);
            if (ImGui::SliderInt("Max bounces", &maxBounces, 1, 16))
                pathTracer.setMaxBounces(maxBounces);
        }

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

        ptLight->visible = renderMode != 1;

        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        controls.update();

        if (!raster) {
            pathTracer.render(scene, camera);
        } else {
            renderer.render(scene, camera);
        }

        ui.render();
    });
}
