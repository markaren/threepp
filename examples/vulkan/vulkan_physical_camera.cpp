// Vulkan physical camera + PostComposite showcase.
//
// The whole scene is authored in REAL photometric units:
//   • sun        = DirectionalLight, 100,000 lux (a clear-day sun)
//   • sky fill   = HemisphereLight
//   • streetlamp = PointLight in lumens (a 150 W HPS lamp ≈ 16,000 lm)
//   • display    = emissive panel in nits (phone-in-sunlight ≈ 2,000 nits)
// and exposed by a real camera: EV100 from aperture/shutter/ISO
// (setPhysicalCamera + setCameraExposure). The Day and Night presets swap
// the LIGHT LEVELS and the CAMERA SETTINGS the way a photographer would —
// the materials never change, yet both land correctly exposed. Auto
// exposure (EV compensation around the physical base), white balance,
// AgX/ACES tone mapping and a lift-less saturation/contrast grade round
// out the camera/display pipeline (PostComposite).

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Vulkan - Physical camera", {{"vsync", false}, {"size", WindowSize{1600, 900}}});
    VulkanRenderer renderer(canvas);

    Scene scene;
    scene.background = Color(0.36f, 0.52f, 0.72f);// daylight sky placeholder

    // ── Geometry: ground + roughness ramp + colour checks ───────────────────
    auto groundMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.color(Color(0.42f, 0.40f, 0.36f)).roughness(0.95f));
    auto ground = Mesh::create(PlaneGeometry::create(60.f, 60.f), groundMat);
    ground->rotation.x = -math::PI / 2.f;
    scene.add(ground);

    for (int i = 0; i < 6; ++i) {// dielectric roughness ramp + one chrome ball
        const bool chrome = i == 5;
        auto m = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                      .color(chrome ? Color(0.95f, 0.95f, 0.95f)
                                                                    : Color(0.75f, 0.72f, 0.68f))
                                                      .roughness(chrome ? 0.05f : 0.1f + 0.17f * static_cast<float>(i))
                                                      .metalness(chrome ? 1.f : 0.f));
        auto s = Mesh::create(SphereGeometry::create(0.5f, 48, 32), m);
        s->position.set(-3.5f + 1.4f * static_cast<float>(i), 0.5f, 0.f);
        scene.add(s);
    }
    const Color checkCols[4] = {Color(0.72f, 0.12f, 0.10f), Color(0.10f, 0.60f, 0.16f),
                                Color(0.10f, 0.22f, 0.75f), Color(0.85f, 0.72f, 0.12f)};
    for (int i = 0; i < 4; ++i) {// colour-checker boxes (WB / grade reference)
        auto m = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(checkCols[i]).roughness(0.7f));
        auto b = Mesh::create(BoxGeometry::create(0.7f, 0.7f, 0.7f), m);
        b->position.set(-2.1f + 1.4f * static_cast<float>(i), 0.35f, 2.0f);
        scene.add(b);
    }

    // ── Lights in real units ─────────────────────────────────────────────────
    auto sun = DirectionalLight::create(Color(1.0f, 0.96f, 0.90f), 100000.f);// lux
    sun->position.set(18.f, 30.f, 12.f);
    scene.add(sun);
    auto sky = HemisphereLight::create(Color(0.55f, 0.70f, 1.00f), Color(0.35f, 0.32f, 0.28f), 14000.f);
    scene.add(sky);

    // 150 W high-pressure-sodium streetlamp ≈ 16,000 lm (converted to candela
    // at upload by setPhysicalLightUnits). Off during the Day preset.
    auto lamp = PointLight::create(Color(1.0f, 0.75f, 0.45f), 0.f);
    lamp->position.set(2.5f, 4.0f, -1.5f);
    lamp->radius = 0.1f;
    scene.add(lamp);
    auto lampBulbMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color::black)
                                                            .emissive(Color(1.0f, 0.75f, 0.45f))
                                                            .roughness(1.f));
    lampBulbMat->emissiveIntensity = 0.f;// nits; lit only at night
    auto lampBulb = Mesh::create(SphereGeometry::create(0.12f, 24, 16), lampBulbMat);
    lampBulb->position.copy(lamp->position);
    scene.add(lampBulb);

    // A 2,000-nit "phone display" panel — barely readable at noon, glaring at
    // night, with NO intensity hacks: nits are nits, the camera does the rest.
    auto panelMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                         .color(Color::black)
                                                         .emissive(Color(0.25f, 0.75f, 1.0f))
                                                         .roughness(0.9f));
    panelMat->emissiveIntensity = 2000.f;// nits
    auto panel = Mesh::create(BoxGeometry::create(1.1f, 0.65f, 0.05f), panelMat);
    panel->position.set(3.2f, 1.0f, 1.2f);
    panel->rotation.y = -0.5f;
    scene.add(panel);

    // ── Physical camera ──────────────────────────────────────────────────────
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.setPhysicalLightUnits(true);
    renderer.setPhysicalCamera(true);
    float aperture = 16.f, shutterInv = 125.f, iso = 100.f;// sunny-16 daylight
    renderer.setCameraExposure(aperture, 1.f / shutterInv, iso);

    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 200.f);
    camera.position.set(0.f, 2.6f, 8.5f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 0.7f, 0.f);
    controls.update();

    // ── UI state ─────────────────────────────────────────────────────────────
    bool physCam = true, autoExp = false;
    float evComp = 0.f;
    float wbTemp = 6500.f, wbTint = 0.f;
    float gradeSat = 1.f, gradeContrast = 1.f;
    int tonemapIdx = 0;// ACES
    const ToneMapping tonemaps[4] = {ToneMapping::ACESFilmic, ToneMapping::AgX,
                                     ToneMapping::Neutral, ToneMapping::None};

    auto applyGrade = [&] {
        VulkanRendererCore::ColorGrade g;
        g.saturation = gradeSat;
        g.contrast = gradeContrast;
        renderer.setColorGrade(g);
    };
    auto applyExposure = [&] { renderer.setCameraExposure(aperture, 1.f / shutterInv, iso); };

    auto dayPreset = [&] {
        sun->intensity = 100000.f;                 // lux
        sky->intensity = 14000.f;
        lamp->intensity = 0.f;
        lampBulbMat->emissiveIntensity = 0.f;
        lampBulbMat->needsUpdate();
        scene.background = Color(0.36f, 0.52f, 0.72f);
        aperture = 16.f; shutterInv = 125.f; iso = 100.f;// sunny 16
        applyExposure();
    };
    auto nightPreset = [&] {
        sun->intensity = 0.f;
        sky->intensity = 40.f;                     // deep-dusk skylight
        lamp->intensity = 16000.f;                 // lumens (150 W HPS)
        lampBulbMat->emissiveIntensity = 30000.f;  // nits — the arc tube itself
        lampBulbMat->needsUpdate();
        scene.background = Color(0.012f, 0.016f, 0.03f);
        aperture = 2.0f; shutterInv = 30.f; iso = 1600.f;// night handheld
        applyExposure();
    };

    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({360, 0});
        ImGui::Begin("Physical camera");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();

        ImGui::TextWrapped("Lights are authored in real units (sun 100k lux, lamp "
                           "16k lm, display 2k nits); the camera exposes them.");
        if (ImGui::Button("Day (sunny 16)")) dayPreset();
        ImGui::SameLine();
        if (ImGui::Button("Night (f/2, 1/30, ISO 1600)")) nightPreset();
        ImGui::Separator();

        if (ImGui::Checkbox("Physical camera", &physCam))
            renderer.setPhysicalCamera(physCam);
        const float ev100 = std::log2(aperture * aperture * shutterInv * 100.f / iso) - evComp;
        ImGui::SameLine();
        ImGui::TextDisabled("EV100 %.1f", ev100);

        bool expDirty = false;
        expDirty |= ImGui::SliderFloat("Aperture (f/)", &aperture, 1.2f, 22.f, "f/%.1f",
                                       ImGuiSliderFlags_Logarithmic);
        expDirty |= ImGui::SliderFloat("Shutter (1/s)", &shutterInv, 8.f, 8000.f, "1/%.0f s",
                                       ImGuiSliderFlags_Logarithmic);
        expDirty |= ImGui::SliderFloat("ISO", &iso, 50.f, 12800.f, "%.0f",
                                       ImGuiSliderFlags_Logarithmic);
        if (expDirty) applyExposure();
        if (ImGui::SliderFloat("EV comp", &evComp, -3.f, 3.f, "%+.1f EV"))
            renderer.setExposureCompensation(evComp);
        if (ImGui::Checkbox("Auto exposure (EV comp around base)", &autoExp))
            renderer.setAutoExposure(autoExp);

        ImGui::Separator();
        ImGui::TextDisabled("Display pipeline");
        const char* tmNames[4] = {"ACES Filmic", "AgX", "Khronos Neutral", "None"};
        if (ImGui::Combo("Tone map", &tonemapIdx, tmNames, 4))
            renderer.toneMapping = tonemaps[tonemapIdx];
        if (ImGui::SliderFloat("WB temp (K)", &wbTemp, 2500.f, 12000.f, "%.0f K",
                               ImGuiSliderFlags_Logarithmic))
            renderer.setWhiteBalance(wbTemp, wbTint);
        if (ImGui::SliderFloat("WB tint", &wbTint, -1.f, 1.f, "%.2f"))
            renderer.setWhiteBalance(wbTemp, wbTint);
        if (ImGui::SliderFloat("Saturation", &gradeSat, 0.f, 2.f, "%.2f")) applyGrade();
        if (ImGui::SliderFloat("Contrast", &gradeContrast, 0.5f, 1.5f, "%.2f")) applyGrade();

        ImGui::Separator();
        ImGui::TextDisabled("Drag = orbit, scroll = zoom");
        ImGui::End();
    });

    IOCapture ioCapture;
    ioCapture.preventMouseEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
    ioCapture.preventScrollEvent = []() -> bool { return ImGui::GetIO().WantCaptureMouse; };
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
