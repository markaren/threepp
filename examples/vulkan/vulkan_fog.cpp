// Vulkan PT — fog (homogeneous participating media) showcase.
// Cornell-style room with a SpotLight aimed across the camera so god rays
// form prominently when fog is enabled. Mirrors examples/wgpu/wgpu_cornell_box.cpp's
// fog UI surface (FogExp2 density + sRGB tint + Henyey-Greenstein anisotropy).
//
// Demonstrates:
//   • Beer-Lambert primary-ray transmittance
//   • Single-scattering inscatter ("god rays") via NEE in raygen
//   • Per-light shadow-ray attenuation in closest_hit
//   • Anisotropy slider sweeping the HG phase from backscatter (-) to
//     forward-scatter (+) god rays.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/threepp.hpp"

using namespace threepp;

namespace {

    auto whiteMat(float roughness = 0.95f) {
        return MeshStandardMaterial::create({{"color", Color(0.73f, 0.73f, 0.73f)}, {"roughness", roughness}});
    }

    // Bright ceiling panel — primary emissive area light.
    auto makeLightPanel() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color::white},
                {"emissive", Color::white},
                {"emissiveIntensity", 18.0f},
                {"roughness", 1.0f},
        });
        auto mesh = Mesh::create(PlaneGeometry::create(2.6f, 2.6f), mat);
        mesh->rotation.x = math::PI / 2.f;
        mesh->position.set(0.f, 9.99f, 0.f);
        return mesh;
    }

    auto makeRoom() {
        auto group = Group::create();
        constexpr float S = 10.f;

        auto floor = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        floor->rotation.x = -math::PI / 2.f;
        group->add(floor);

        auto ceiling = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        ceiling->rotation.x = math::PI / 2.f;
        ceiling->position.y = S;
        group->add(ceiling);

        auto back = Mesh::create(PlaneGeometry::create(S, S), whiteMat());
        back->position.set(0.f, S / 2.f, -S / 2.f);
        group->add(back);

        auto leftMat = MeshStandardMaterial::create({{"color", Color(0.65f, 0.05f, 0.05f)}, {"roughness", 0.95f}});
        auto left = Mesh::create(PlaneGeometry::create(S, S), leftMat);
        left->rotation.y = math::PI / 2.f;
        left->position.set(-S / 2.f, S / 2.f, 0.f);
        group->add(left);

        auto rightMat = MeshStandardMaterial::create({{"color", Color(0.12f, 0.45f, 0.15f)}, {"roughness", 0.95f}});
        auto right = Mesh::create(PlaneGeometry::create(S, S), rightMat);
        right->rotation.y = -math::PI / 2.f;
        right->position.set(S / 2.f, S / 2.f, 0.f);
        group->add(right);

        return group;
    }

    auto makeTallBox() {
        auto mesh = Mesh::create(BoxGeometry::create(1.5f, 3.f, 1.5f), whiteMat());
        mesh->position.set(-1.5f, 1.5f, -1.5f);
        mesh->rotation.y = 0.3f;
        return mesh;
    }

    auto makeShortBox() {
        auto mesh = Mesh::create(BoxGeometry::create(1.5f, 1.5f, 1.5f), whiteMat());
        mesh->position.set(1.5f, 0.75f, 1.0f);
        mesh->rotation.y = -0.3f;
        return mesh;
    }

    auto makeMetalSphere() {
        auto mat = MeshStandardMaterial::create({
                {"color", Color(0.95f, 0.93f, 0.88f)},
                {"roughness", 0.04f},
                {"metalness", 1.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.6f, 48, 48), mat);
        mesh->position.set(1.5f, 2.1f, 1.0f);
        return mesh;
    }

    auto makeGlassSphere() {
        auto mat = MeshPhysicalMaterial::create({
                {"color", Color::white},
                {"transmission", 0.95f},
                {"ior", 1.5f},
                {"roughness", 0.0f},
                {"metalness", 0.0f},
        });
        auto mesh = Mesh::create(SphereGeometry::create(0.7f, 48, 48), mat);
        mesh->position.set(-1.5f, 0.7f, 1.8f);
        return mesh;
    }

}// namespace

int main() {

    Canvas canvas("Vulkan PT - Fog", {{"vsync", false}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 0.9f;

    Scene scene;
    scene.background = Color::black;

    scene.add(makeRoom());
    scene.add(makeLightPanel());
    scene.add(makeTallBox());
    scene.add(makeShortBox());
    scene.add(makeMetalSphere());
    scene.add(makeGlassSphere());

    // SpotLight aimed across the camera ray for prominent god rays. The cone
    // angle and falloff are tuned so the light shaft stays mostly inside the
    // room volume — a too-wide cone washes out the shaft.
    auto spot = SpotLight::create(Color(1.0f, 0.9f, 0.7f), 80.0f);
    spot->position.set(-3.5f, 8.0f, 3.0f);
    Object3D spotTarget;
    spotTarget.position.set(2.0f, 0.5f, -2.0f);
    spot->setTarget(spotTarget);
    spot->angle = 0.35f;
    spot->penumbra = 0.4f;
    spot->distance = 25.0f;
    spot->decay = 2.0f;
    scene.add(spot);

    PerspectiveCamera camera(50.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 5.f, 14.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 4.f, 0.f);
    controls.update();

    bool fogOn = true;
    float fogDensity = 0.08f;
    float fogColor[3] = {0.55f, 0.55f, 0.62f};
    float fogG = 0.6f;// HG anisotropy: forward-scattering god rays by default
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({320, 0});
        ImGui::Begin("Vulkan PT - Fog");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();

        ImGui::TextWrapped("FogExp2 single-scattering volumetrics. Toggle off "
                           "to compare. Anisotropy g sweeps the HG phase from "
                           "backscatter (-) to forward god rays (+).");
        ImGui::Separator();

        ImGui::Checkbox("Fog", &fogOn);
        if (fogOn) {
            ImGui::SliderFloat("Density", &fogDensity, 0.001f, 0.5f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::ColorEdit3("Color", fogColor);
            ImGui::SliderFloat("Anisotropy g", &fogG, -0.9f, 0.9f, "%.2f");
        }

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

        if (fogOn) {
            scene.fog = FogExp2(Color(fogColor[0], fogColor[1], fogColor[2]), fogDensity);
            renderer.setFogAnisotropy(fogG);
        } else {
            scene.fog.reset();
        }

        renderer.render(scene, camera);
        ui.render();
    });

    return 0;
}
