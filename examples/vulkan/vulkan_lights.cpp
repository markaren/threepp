// Vulkan PT lights showcase — five side-by-side sections each lit by a
// different light source, demonstrating the renderer's analytic light path
// (DirectionalLight, PointLight, SpotLight, RectAreaLight) and the emissive-
// mesh NEE path. AmbientLight contributes a uniform diffuse fill that the
// user can toggle to see the shadow falloff with no fill.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/RectAreaLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    // Five sections side-by-side, 3.4 units wide each.
    constexpr float kSectionDx = 3.4f;
    constexpr float kSectionXs[5] = {
            -2 * kSectionDx,
            -1 * kSectionDx,
            0.0f,
            +1 * kSectionDx,
            +2 * kSectionDx,
    };
    constexpr float kRoomDepth = 6.0f;
    constexpr float kRoomHeight = 5.0f;

    auto matteWhite() {
        return MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.78f, 0.78f, 0.78f)).roughness(0.9f));
    }

    void addRoom(Scene& scene) {
        const float roomWidth = 5 * kSectionDx + 1.0f;
        // Floor
        auto floor = Mesh::create(PlaneGeometry::create(roomWidth, kRoomDepth), matteWhite());
        floor->rotation.x = -math::PI / 2.f;
        scene.add(floor);
        // Back wall
        auto back = Mesh::create(PlaneGeometry::create(roomWidth, kRoomHeight), matteWhite());
        back->position.set(0.f, kRoomHeight / 2.f, -kRoomDepth / 2.f);
        scene.add(back);
        // Ceiling — kept dark + matte so emissive panels above it don't lift
        // the unlit sections; lighting differences between sections stay visible.
        auto ceil = Mesh::create(PlaneGeometry::create(roomWidth, kRoomDepth),
                                 MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.15f, 0.15f, 0.15f)).roughness(1.0f)));
        ceil->rotation.x = math::PI / 2.f;
        ceil->position.y = kRoomHeight;
        scene.add(ceil);

        // Dividing walls between each pair of sections — keep each light's
        // contribution local. Two-sided so the camera can pan around either
        // side of a divider without seeing through it.
        for (int i = 0; i + 1 < 5; ++i) {
            const float wx = 0.5f * (kSectionXs[i] + kSectionXs[i + 1]);
            auto wallMat = matteWhite();
            wallMat->side = Side::Double;
            auto wall = Mesh::create(PlaneGeometry::create(kRoomDepth, kRoomHeight), wallMat);
            wall->rotation.y = math::PI / 2.f;
            wall->position.set(wx, kRoomHeight / 2.f, 0.f);
            scene.add(wall);
        }
    }

    // Per-section receivers: a chrome sphere (highlights), a matte sphere
    // (diffuse + shadow), and a small cube (cast a hard shadow on the floor).
    void addReceivers(Scene& scene, float sectionX) {
        auto chromeMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.95f, 0.95f, 0.95f)).roughness(0.05f).metalness(1.0f));
        auto chrome = Mesh::create(SphereGeometry::create(0.45f, 48, 32), chromeMat);
        chrome->position.set(sectionX - 0.85f, 0.45f, 0.6f);
        scene.add(chrome);

        auto matte = Mesh::create(SphereGeometry::create(0.45f, 48, 32), matteWhite());
        matte->position.set(sectionX + 0.0f, 0.45f, 0.0f);
        scene.add(matte);

        auto cubeMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.85f, 0.85f, 0.85f)).roughness(0.6f));
        auto cube = Mesh::create(BoxGeometry::create(0.55f, 0.55f, 0.55f), cubeMat);
        cube->position.set(sectionX + 0.85f, 0.275f, 0.5f);
        cube->rotation.y = 0.4f;
        scene.add(cube);
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT - Lights", {{"vsync", false}, {"size", WindowSize{1700, 900}}});
    VulkanRenderer renderer(canvas);

    Scene scene;
    scene.background = Color(0.02f, 0.02f, 0.02f);

    addRoom(scene);
    for (float sx : kSectionXs) addReceivers(scene, sx);

    // ── Section 1: DirectionalLight ─────────────────────────────────────────
    // Sun-like — direction is what matters; position only affects the shadow
    // camera, not the lighting. Warm color, parallel rays.
    auto dirLight = DirectionalLight::create(Color(1.0f, 0.85f, 0.65f), 2.5f);
    dirLight->position.set(kSectionXs[0] + 2.f, 4.f, 2.f);
    Object3D dirlightTarget;
    dirlightTarget.position.set(kSectionXs[0], 0.f, 0.f);
    dirLight->setTarget(dirlightTarget);
    scene.add(dirLight);

    // ── Section 2: PointLight ───────────────────────────────────────────────
    // Omni-directional with inverse-square falloff. Cool color so it visually
    // separates from the warm DirectionalLight neighbour.
    auto pointLight = PointLight::create(Color(0.55f, 0.75f, 1.0f), 12.0f);
    pointLight->position.set(kSectionXs[1], 1.6f, 0.5f);
    scene.add(pointLight);

    // ── Section 3: SpotLight ────────────────────────────────────────────────
    // Cone-shaped beam, sharp penumbra controlled by penumbra ∈ [0, 1].
    // Magenta so it reads as a stage spotlight. Aimed at the floor below.
    auto spotLight = SpotLight::create(Color(1.0f, 0.35f, 0.9f), 30.0f);
    Object3D spotlightTarget;
    spotlightTarget.position.set(kSectionXs[2], 0.f, 0.f);
    spotLight->position.set(kSectionXs[2], 3.5f, 0.5f);
    spotLight->setTarget(spotlightTarget);
    spotLight->angle = math::PI / 7.f;
    spotLight->penumbra = 0.4f;
    spotLight->distance = 0.f;// 0 = infinite range
    scene.add(spotLight);

    // ── Section 4: RectAreaLight ────────────────────────────────────────────
    // Soft area light — produces realistic soft shadows at sphere edges and
    // contact-shadow penumbra under the cube. Faces -Z by default; rotate to
    // face down. Warm white, modest size so the demo room can fit it.
    auto rectLight = RectAreaLight::create(Color(1.0f, 0.95f, 0.85f), 8.0f, 1.6f, 0.6f);
    rectLight->position.set(kSectionXs[3], 3.2f, 0.3f);
    // lookAt-target directly below the light so its local -Z (emit direction)
    // points straight down; the panel visualization matches with rotation.x.
    rectLight->lookAt(Vector3(kSectionXs[3], 0.f, 0.3f));
    scene.add(rectLight);
    // Visualize the panel — emissive plane matched to the light dimensions.
    // PlaneGeometry's front face is +Z, so rotation.x = -π/2 turns it face-down.
    auto rectPanelMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::black).emissive(Color(1.0f, 0.95f, 0.85f)).emissiveIntensity(6.0f).roughness(1.0f));
    auto rectPanel = Mesh::create(PlaneGeometry::create(1.6f, 0.6f), rectPanelMat);
    rectPanel->position.copy(rectLight->position);
    rectPanel->rotation.x = -math::PI / 2.f;
    scene.add(rectPanel);

    // ── Section 5: Emissive mesh (NEE-sampled) ──────────────────────────────
    // No analytic light here — illumination comes purely from a glowing torus
    // that the PT samples explicitly via emissive-triangle NEE. Cyan to make
    // the colour bleed obvious on the matte receivers.
    auto neonMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::black).emissive(Color(0.1f, 0.95f, 1.0f)).emissiveIntensity(22.0f).roughness(1.0f));
    auto neon = Mesh::create(TorusGeometry::create(0.55f, 0.05f, 16, 64), neonMat);
    neon->rotation.x = math::PI / 2.f;
    neon->position.set(kSectionXs[4], 2.6f, 0.5f);
    scene.add(neon);

    // ── AmbientLight (toggleable fill) ──────────────────────────────────────
    auto ambient = AmbientLight::create(Color(1.0f, 1.0f, 1.0f), 0.0f);
    scene.add(ambient);

    // ── Camera + controls ───────────────────────────────────────────────────
    PerspectiveCamera camera(38.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 3.6f, 11.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 1.f, 0.f);
    controls.update();

    // ── ImGui ───────────────────────────────────────────────────────────────
    bool dirOn = true, pointOn = true, spotOn = true, rectOn = true, neonOn = true;
    float dirIntensity = dirLight->intensity;
    float pointIntensity = pointLight->intensity;
    float spotIntensity = spotLight->intensity;
    float rectIntensity = rectLight->intensity;
    float neonIntensity = neonMat->emissiveIntensity;
    float ambIntensity = 0.0f;

    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({340, 0});
        ImGui::Begin("Vulkan PT - Lights");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();

        ImGui::TextWrapped("Five sections, each lit by a different light type:"
                           " directional, point, spot, rect-area, emissive mesh.");
        ImGui::Separator();

        if (ImGui::Checkbox("Directional (sun)", &dirOn))
            dirLight->intensity = dirOn ? dirIntensity : 0.f;
        ImGui::SameLine();
        if (ImGui::SliderFloat("##diI", &dirIntensity, 0.f, 8.f, "%.1f") && dirOn)
            dirLight->intensity = dirIntensity;

        if (ImGui::Checkbox("Point", &pointOn))
            pointLight->intensity = pointOn ? pointIntensity : 0.f;
        ImGui::SameLine();
        if (ImGui::SliderFloat("##plI", &pointIntensity, 0.f, 60.f, "%.1f") && pointOn)
            pointLight->intensity = pointIntensity;

        if (ImGui::Checkbox("Spot", &spotOn))
            spotLight->intensity = spotOn ? spotIntensity : 0.f;
        ImGui::SameLine();
        if (ImGui::SliderFloat("##slI", &spotIntensity, 0.f, 80.f, "%.1f") && spotOn)
            spotLight->intensity = spotIntensity;

        if (ImGui::Checkbox("Rect area", &rectOn)) {
            rectLight->intensity = rectOn ? rectIntensity : 0.f;
            rectPanel->visible = rectOn;// hide emissive panel so NEE can't sample it
        }
        ImGui::SameLine();
        if (ImGui::SliderFloat("##rlI", &rectIntensity, 0.f, 30.f, "%.1f") && rectOn)
            rectLight->intensity = rectIntensity;

        // Emissive torus is the only light here — toggling intensity to 0 alone
        // doesn't bump the material version, so the GPU MaterialDesc keeps the
        // old emissive and the torus still glows. Hide the mesh instead.
        if (ImGui::Checkbox("Emissive mesh", &neonOn))
            neon->visible = neonOn;
        ImGui::SameLine();
        if (ImGui::SliderFloat("##neI", &neonIntensity, 0.f, 60.f, "%.1f") && neonOn) {
            neonMat->emissiveIntensity = neonIntensity;
            neonMat->needsUpdate();// version bump so PT picks up the new value
        }

        ImGui::Separator();
        if (ImGui::SliderFloat("Ambient fill", &ambIntensity, 0.0f, 1.0f, "%.2f"))
            ambient->intensity = ambIntensity;

        ImGui::Separator();

        ImGui::Separator();
        bool restirDI = renderer.restirDIEnabled();
        if (ImGui::Checkbox("ReSTIR DI", &restirDI))
            renderer.setRestirDIEnabled(restirDI);

        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reservoir-resampled direct lighting at primary.\n"
                              "Off = classic per-light NEE (one shadow ray\n"
                              "per analytic light + emissive sample).");

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
