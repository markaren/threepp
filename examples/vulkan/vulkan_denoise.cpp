// Vulkan PT denoiser showcase — port of examples/wgpu/wgpu_denoise.cpp.
// Rotating torus knot on a pedestal with three colored analytical lights, a
// glass sphere, and a metal sphere. Demonstrates the denoiser, ReSTIR DI,
// and the spec/diff split + demod (step 1.4c) on textured + transmissive
// content under camera and object motion.
//
// Differences vs the WGPU example:
//   • VulkanRenderer is always a path tracer (no `usePathTracer` toggle and
//     no separate `pathTracer()` object — settings live on the renderer).
//   • Max bounce count is not yet plumbed to a setter on Vulkan
//     (compile-time `kMaxBounces = 4` in raygen.rgen), so the slider is
//     omitted.
//   • Vulkan exposes hybrid raster + samples-per-pixel knobs that the WGPU
//     example doesn't — added below for completeness.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanPathTracer.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    // Ground plane — large, slightly rough.
    auto makeGround() {
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.4f, 0.4f, 0.4f))
                .roughness(0.8f)
                .metalness(0.0f));
        auto mesh = Mesh::create(PlaneGeometry::create(20.f, 20.f), mat);
        mesh->rotation.x = -math::PI / 2.f;
        return mesh;
    }

    // Pedestal — cylinder.
    auto makePedestal() {
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.7f, 0.7f, 0.72f))
                .roughness(0.6f)
                .metalness(0.0f)
                .side(Side::Double));
        auto mesh = Mesh::create(CylinderGeometry::create(1.0f, 1.2f, 1.5f, 64), mat);
        mesh->position.set(0.f, 0.75f - 0.05f, 0.f);
        return mesh;
    }

    // Hero object — torus knot. Slightly metallic so it picks up the colored
    // analytical lights without losing diffuse texture.
    auto makeHeroObject() {
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.9f, 0.15f, 0.1f))
                .roughness(0.9f)
                .metalness(0.2f));
        auto mesh = Mesh::create(TorusKnotGeometry::create(0.7f, 0.25f, 128, 32), mat);
        mesh->position.set(0.f, 2.8f, 0.f);
        return mesh;
    }

    // Back wall — polished metal so the scene shows up in reflection.
    auto makeBackWall() {
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.75f, 0.75f, 0.75f))
                .roughness(0.1f)
                .metalness(0.9f));
        auto mesh = Mesh::create(BoxGeometry::create(12.f, 8.f, 0.1f), mat);
        mesh->position.set(0.f, 4.f, -5.f);
        return mesh;
    }

    // Glass sphere beside the pedestal — clear, smooth, IOR 1.5.
    auto makeGlassSphere() {
        auto mat = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}
                .color(Color::steelblue)
                .transmission(0.95f)
                .ior(1.5f)
                .roughness(0.0f)
                .metalness(0.0f));
        auto mesh = Mesh::create(SphereGeometry::create(1.5f, 48, 48), mat);
        mesh->position.set(-4.5f, 1.5f, 1.5f);
        return mesh;
    }

    // Small metal sphere on the other side — tinted chrome.
    auto makeMetalSphere() {
        auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                .color(Color(0.95f, 0.85f, 0.4f))
                .roughness(0.05f)
                .metalness(1.0f));
        auto mesh = Mesh::create(SphereGeometry::create(0.4f, 48, 48), mat);
        mesh->position.set(2.2f, 0.4f, 1.8f);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT - Denoiser Showcase", {{"vsync", false}});

    VulkanPathTracer renderer(canvas);
    renderer.outputColorSpace    = ColorSpace::sRGB;
    renderer.toneMapping         = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    // AAA post stack: punchy HDR bloom + post-TAA RCAS sharpen (live-tunable
    // below). Bloom is additive over a soft-knee bright pass, so darks stay
    // crisp; sharpen restores detail the temporal resolve softens.
    renderer.setBloomIntensity(0.5f);
    renderer.setBloomThreshold(1.0f);
    // Cap the bloom input: the analytic lights mirrored in the smooth gold
    // sphere are sub-pixel HDR spikes whose intensity swings per frame with
    // the TAA jitter — unclamped, the halo radius visibly pulses.
    renderer.setBloomClamp(16.0f);
    renderer.setSharpenStrength(0.5f);

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

    // ---- Analytical lights ----
    // Warm key spotlight from upper right.
    Object3D target;
    target.position.set(0.f, 2.f, 0.f);

    auto keyLight = SpotLight::create(Color(1.0f, 0.9f, 0.7f), 62.0f, 15.f, math::PI / 6.f, 0.3f);
    keyLight->position.set(4.f, 7.f, 3.f);
    keyLight->setTarget(target);
    scene.add(keyLight);

    // Cool fill point light from the left.
    auto fillLight = PointLight::create(Color(0.4f, 0.6f, 1.0f), 6.5f);
    fillLight->position.set(-4.f, 4.f, 2.f);
    scene.add(fillLight);

    // Rim/back warm point light behind.
    auto rimLight = PointLight::create(Color(1.0f, 0.5f, 0.2f), 5.2f);
    rimLight->position.set(1.f, 5.f, -4.f);
    scene.add(rimLight);

    // ---- Camera ----
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 3.5f, 8.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 2.f, 0.f);
    controls.update();

    // ---- UI state ----
    bool denoiserOn   = renderer.denoise();
    bool restirOn     = renderer.restirDIEnabled();
    bool restirGiOn   = renderer.restirGIEnabled();
    bool perSppJitter = renderer.perSppJitterHybrid();
    bool rotating     = true;
    float rotSpeed    = 0.5f;
    float exposure    = renderer.toneMappingExposure;
    int spp           = renderer.samplesPerPixel();
    float bloomInt    = renderer.bloomIntensity();
    float bloomThresh = renderer.bloomThreshold();
    float bloomClamp  = renderer.bloomClamp();
    float sharpen     = renderer.sharpenStrength();
    uint64_t frames   = 0;
    float fps         = 0.f;
    float fpsAccum    = 0.f;
    int fpsFrames     = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
        ImGui::SetNextWindowSize({320, 0}, ImGuiCond_Once);
        ImGui::Begin("Vulkan PT - Denoiser");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frames: %llu", static_cast<unsigned long long>(frames));
        ImGui::Separator();

        ImGui::Checkbox("Rotate object", &rotating);
        if (rotating) {
            ImGui::SliderFloat("Speed", &rotSpeed, 0.0f, 3.0f);
        }
        ImGui::Separator();

        if (ImGui::Checkbox("Denoiser", &denoiserOn))
            renderer.setDenoise(denoiserOn);
        if (ImGui::Checkbox("ReSTIR DI", &restirOn))
            renderer.setRestirDIEnabled(restirOn);
        if (ImGui::Checkbox("ReSTIR GI", &restirGiOn))
            renderer.setRestirGIEnabled(restirGiOn);
        if (ImGui::Checkbox("Per-spp AA jitter", &perSppJitter))
            renderer.setPerSppJitterHybrid(perSppJitter);

        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f))
            renderer.toneMappingExposure = exposure;
        if (ImGui::SliderInt("Samples / pixel", &spp, 1, 8))
            renderer.setSamplesPerPixel(spp);

        ImGui::Separator();
        ImGui::TextDisabled("AAA post");
        if (ImGui::SliderFloat("Bloom intensity", &bloomInt, 0.0f, 1.5f))
            renderer.setBloomIntensity(bloomInt);
        if (ImGui::SliderFloat("Bloom threshold", &bloomThresh, 0.0f, 3.0f))
            renderer.setBloomThreshold(bloomThresh);
        if (ImGui::SliderFloat("Bloom clamp", &bloomClamp, 0.0f, 64.0f))
            renderer.setBloomClamp(bloomClamp);
        if (ImGui::SliderFloat("Sharpen (RCAS)", &sharpen, 0.0f, 0.8f))
            renderer.setSharpenStrength(sharpen);

        ImGui::Separator();
        const auto t = renderer.lastFrameTimings();
        ImGui::Text("GPU: PT %.2f ms  Denoise %.2f ms", t.pathTraceMs, t.denoiseMs);
        ImGui::Text("     raster %.2f ms  TAA %.2f ms", t.rasterGbufMs, t.taaMs);

        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent    = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent   = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventKeyboardEvent = []() -> bool { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&ioCapture);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        camera.aspect = canvas.aspect();
        camera.updateProjectionMatrix();
    });

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();

        if (rotating) {
            hero->rotation.y += rotSpeed * dt;
        }

        ++frames;
        fpsAccum += dt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        controls.update();
        renderer.render(scene, camera);
        ui.render();
    });

    return 0;
}
