// Deferred-renderer denoiser / ReSTIR DI showcase. Rotating torus knot on a
// pedestal with three colored analytical lights, a glass sphere, and a metal
// sphere. Demonstrates the deferred VulkanRenderer's SVGF-style filtering of
// the demodulated ray-traced lighting channels (`setDenoise`), ReSTIR DI, and
// the spec/diff split + demod on textured + transmissive content under camera
// and object motion.
//
// The deferred VulkanRenderer runs its filter + composite stage over the
// ray-traced lighting channels (`setDenoise`); settings live directly on the
// renderer. An AAA post stack (HDR bloom, RCAS sharpen, motion blur) rides on
// top — surfaced below for completeness.

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/geometries/TorusKnotGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/threepp.hpp"

#include "capture_util.hpp"
#include "window_util.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

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

int main(int argc, char** argv) {

    // Dev capture: --shot <name.png> [--frames N] settles N frames then writes
    // aaa_caps/<name> and exits (object rotation off for a clean settle).
    // --lightrad r sets a physical source radius on the key/fill/rim lights —
    // soft local-shadow triage for the deferred denoised-shadow channel.
    // --shot / --frames are the shared capture flags (150-frame settle here);
    // everything below is specific to this demo's triage knobs.
    capture::Shot shot(capture::parseArgs(argc, argv), /*defaultFrames=*/150);
    float optLightRad = -1.f;
    float fogDensity  = 0.f;// --fog d: FogExp2 medium (froxel-volumetrics triage: beams + point-light glow)
    int   ringLights  = 0;// --lights N: ring of N extra colored point lights (clustered-lighting triage)
    float mblur       = 0.f;// --mblur s: post-TAA motion blur, shutter fraction (0.5 = 180°).
                            // Keeps the hero rotating in shot mode so the streaks show.
    float optSharpen  = -1.f;// --sharpen s: RCAS override (0 exercises the blur→swapchain chain)
    float stepRad     = -1.f;// --radstep R N: set light radius to R at shot frame N —
    int   stepFrame   = -1;  // the interactive slider-drag case, for settle-speed measurement
    int   seqN        = 0;   // --seqn K: write K CONSECUTIVE frames (stem_000.png …) from
                             // --frames on — frame-to-frame diffs measure settle churn
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--lightrad" && i + 1 < argc) optLightRad = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--lights" && i + 1 < argc) ringLights = std::atoi(argv[++i]);
        else if (a == "--fog" && i + 1 < argc) fogDensity = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--mblur" && i + 1 < argc) mblur = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--sharpen" && i + 1 < argc) optSharpen = static_cast<float>(std::atof(argv[++i]));
        else if (a == "--radstep" && i + 2 < argc) {
            stepRad   = static_cast<float>(std::atof(argv[++i]));
            stepFrame = std::atoi(argv[++i]);
        }
        else if (a == "--seqn" && i + 1 < argc) seqN = std::atoi(argv[++i]);
    }

    Canvas canvas("Vulkan Deferred - Denoiser Showcase", {{"vsync", false}});

    VulkanRenderer renderer(canvas);
    renderer.outputColorSpace    = ColorSpace::sRGB;

    // AAA post stack: punchy HDR bloom + post-TAA RCAS sharpen (live-tunable
    // below). Bloom is additive over a soft-knee bright pass, so darks stay
    // crisp; sharpen restores detail the temporal resolve softens.
    renderer.setBloomIntensity(0.5f);
    renderer.setBloomThreshold(1.0f);
    // Cap the bloom input: the analytic lights mirrored in the smooth gold
    // sphere are sub-pixel HDR spikes whose intensity swings per frame with
    // the TAA jitter — unclamped, the halo radius visibly pulses.
    renderer.setBloomClamp(16.0f);
    renderer.setSharpenStrength(optSharpen >= 0.f ? optSharpen : 0.5f);
    if (mblur > 0.f) renderer.setMotionBlur(mblur);

    // ---- Scene ----
    Scene scene;
    scene.background = Color(0.02f, 0.02f, 0.05f);
    if (fogDensity > 0.f) {// --fog: key-spot beam + ring-light glow through the medium
        scene.fog = FogExp2(Color(0.35f, 0.38f, 0.45f), fogDensity);
        renderer.setFogAnisotropy(0.4f);
        renderer.setVolumetricFog(true);
    }

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

    if (optLightRad >= 0.f) {// --lightrad: soft local shadows (source radius)
        keyLight->radius  = optLightRad;
        fillLight->radius = optLightRad;
        rimLight->radius  = optLightRad;
    }

    // --lights N: ring of N colored point lights around the set with unlit
    // marker spheres — clustered-lighting triage. The pre-cluster renderer
    // capped point lights at 8 (the rest were silently dropped); with the
    // cluster path every one of them lights its own pool of floor.
    for (int i = 0; i < ringLights; ++i) {
        const float a = static_cast<float>(i) / static_cast<float>(ringLights) * 6.2831853f;
        Color c;
        c.setHSL(static_cast<float>(i) / static_cast<float>(ringLights), 0.9f, 0.5f);
        auto l = PointLight::create(c, 4.f, 7.f, 2.f);
        l->position.set(std::cos(a) * 6.5f, 0.9f, std::sin(a) * 6.5f);
        if (optLightRad >= 0.f) l->radius = optLightRad;
        scene.add(l);
        auto marker = Mesh::create(SphereGeometry::create(0.09f),
                                   MeshBasicMaterial::create(MeshBasicMaterial::Params{}.color(c)));
        marker->position.copy(l->position);
        scene.add(marker);
    }

    // ---- Camera ----
    PerspectiveCamera camera(45.f, canvas.aspect(), 0.1f, 100.f);
    camera.position.set(0.f, 3.5f, 8.f);
    OrbitControls controls{camera, canvas};
    controls.target.set(0.f, 2.f, 0.f);
    controls.update();

    // ---- UI state ----
    bool rotating  = true;
    float rotSpeed = 0.5f;

    // Runtime renderer settings (shared panel: denoiser, bloom, sharpen,
    // motion blur, timings, ...). Interactive runs only — the headless capture
    // path must not draw UI into the measured frames.
    std::unique_ptr<RendererSettingsUi> ui;
    if (!shot.active()) {
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            ImGui::Checkbox("Rotate object", &rotating);
            if (rotating) {
                ImGui::SliderFloat("Speed", &rotSpeed, 0.0f, 3.0f);
            }
        }, "Vulkan Deferred - Denoiser");
    }

    demo::bindResize(canvas, renderer, camera);

    Clock clock;

    canvas.animate([&] {
        const float dt = clock.getDelta();

        // Shots settle with the object still — EXCEPT the motion-blur triage,
        // which needs per-object motion vectors in the capture frame.
        if (rotating && (!shot.active() || mblur > 0.f)) {
            hero->rotation.y += rotSpeed * dt;
        }

        if (shot.active() && stepFrame >= 0 && shot.frame == stepFrame) {
            keyLight->radius  = stepRad;// --radstep: mid-run light edit (settle harness)
            fillLight->radius = stepRad;
            rimLight->radius  = stepRad;
        }
        controls.update();
        renderer.render(scene, camera);
        if (!shot.active()) {
            ui->render();
        } else if (shot.ready()) {
            if (seqN > 0) {// --seqn: consecutive frames stem_000.png … (settle-churn metric)
                const int k = shot.frame - shot.frames;
                const auto p = capture::writeShotSequenceFrame(renderer, shot.name, k);
                std::cout << "wrote " << p.string() << std::endl;
                if (k + 1 >= seqN) std::exit(0);
            } else {
                capture::finishShot(renderer, shot.name);
            }
        }
    });

    return 0;
}
