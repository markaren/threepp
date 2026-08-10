// Vulkan deferred — campfire at night (plans/particle-atmosphere.md F1).
//
// The whole demo is one FireEffect and some things for it to light. That is
// the point: the flame is a ParticleField whose density carries F0's blackbody
// emission ramp, the smoke is a second field with its own albedo, and the
// warm moving light on the stones, the logs and the ground is an ordinary
// PointLight — so every renderer feature on screen already existed. Nothing
// here reaches into the backend.
//
// What to look for:
//   • a flame whose SHAPE is particles, not a texture — no billboard cards
//   • flickering warm light with MOVING ray-traced shadows on the props
//   • a smoke column leaning away on the wind, unlit and grey, right above a
//     flame that is emissive and warm — two media, two albedos, one scene
//     (the reason F0 made the medium params per field)
//   • embers rising and burning out
//   • bloom on the core, and the froxel glow of the firelight in the height fog
//
// Controls:  drag = orbit   scroll = zoom   SPACE = ignite / extinguish
// Headless:  vulkan_fire --shot out.png [--frames N] [--t SECONDS] [--no-embers]
//            vulkan_fire --bench            interleaved A/B of the effect's GPU cost

#include "capture_util.hpp"

#include "threepp/extras/effects/FireEffect.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/IcosahedronGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kTau = 6.28318530718f;

    auto matteMat(const Color& c, float rough = 0.92f) {
        return MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(c).roughness(rough).metalness(0.f));
    }

    // A deterministic little hash, so the scene dressing is the same picture
    // every run — a capture that has to be LOOKED at and compared is worthless
    // if the rocks move between builds.
    float h01(std::uint32_t i, std::uint32_t s) {
        std::uint32_t x = i * 0x9e3779b9u + s * 0x85ebca6bu;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        return float(x >> 8) * (1.f / 16777216.f);
    }

    // ── The fire ring ───────────────────────────────────────────────────────
    // Stones close in are the best light-catchers in the scene: they sit inside
    // the inverse-square falloff, so the flicker reads on them hardest, and
    // they throw the short moving shadows that say "point source, right there".
    std::shared_ptr<Group> makeStoneRing(float radius) {
        auto g = Group::create();
        constexpr int kStones = 11;
        for (int i = 0; i < kStones; ++i) {
            const float a = kTau * (float(i) + 0.35f * h01(std::uint32_t(i), 7u)) / float(kStones);
            const float sc = 0.10f + 0.06f * h01(std::uint32_t(i), 11u);
            auto m = Mesh::create(IcosahedronGeometry::create(sc, 1),
                                  matteMat(Color(0.30f + 0.08f * h01(std::uint32_t(i), 13u),
                                                 0.28f, 0.26f),
                                           0.95f));
            m->position.set(radius * std::cos(a), sc * 0.55f, radius * std::sin(a));
            m->rotation.set(h01(std::uint32_t(i), 17u) * kTau,
                            h01(std::uint32_t(i), 19u) * kTau,
                            h01(std::uint32_t(i), 23u) * kTau);
            m->scale.set(1.f, 0.72f, 1.15f);
            g->add(m);
        }
        return g;
    }

    // Logs leaning into the fire — the fuel, and the geometry the flame is
    // actually standing on, so the emissive volume has something solid at its
    // base instead of hanging in the air.
    std::shared_ptr<Group> makeFuel() {
        auto g = Group::create();
        auto wood = matteMat(Color(0.16f, 0.10f, 0.07f), 0.95f);
        for (int i = 0; i < 5; ++i) {
            const float a = kTau * float(i) / 5.f + 0.3f;
            auto log = Mesh::create(CylinderGeometry::create(0.035f, 0.045f, 0.72f, 10), wood);
            log->position.set(0.16f * std::cos(a), 0.16f, 0.16f * std::sin(a));
            log->rotation.set(0.f, -a, 1.05f);
            g->add(log);
        }
        // A charred bed under them, so the ground right at the fire is not a
        // flat plane the light bounces off unnaturally evenly.
        auto bed = Mesh::create(CylinderGeometry::create(0.34f, 0.40f, 0.05f, 20),
                                matteMat(Color(0.05f, 0.045f, 0.04f), 1.f));
        bed->position.y = 0.025f;
        g->add(bed);
        return g;
    }

    // Things at a few metres: they catch much less light, which is exactly what
    // makes the falloff read. The stack of firewood and the two posts also give
    // the moving shadows something long to sweep across the ground.
    std::shared_ptr<Group> makeCamp() {
        auto g = Group::create();
        auto wood = matteMat(Color(0.19f, 0.13f, 0.09f), 0.9f);

        auto bench = Mesh::create(BoxGeometry::create(1.9f, 0.13f, 0.34f), wood);
        bench->position.set(0.15f, 0.36f, -1.75f);
        bench->rotation.y = 0.12f;
        g->add(bench);
        for (float dx : {-0.75f, 0.95f}) {
            auto leg = Mesh::create(BoxGeometry::create(0.13f, 0.36f, 0.30f), wood);
            leg->position.set(0.15f + dx, 0.18f, -1.75f);
            g->add(leg);
        }

        // Split logs stacked beside the fire.
        for (int i = 0; i < 9; ++i) {
            const int row = i / 3, col = i % 3;
            auto log = Mesh::create(CylinderGeometry::create(0.055f, 0.055f, 0.62f, 10), wood);
            log->position.set(1.55f + 0.02f * float(row),
                              0.06f + 0.115f * float(row),
                              -0.55f + 0.125f * float(col) - 0.03f * float(row));
            log->rotation.set(0.f, 0.18f, math::PI / 2.f);
            g->add(log);
        }

        // Two posts with a slack line — vertical geometry catching a rim of
        // firelight, and the tallest thing near enough to be lit at all.
        for (float dx : {-1.9f, -1.15f}) {
            auto post = Mesh::create(CylinderGeometry::create(0.05f, 0.06f, 1.6f, 10), wood);
            post->position.set(dx, 0.8f, 0.9f);
            g->add(post);
        }

        // A packed bag and a pot: small, close, curved — the shapes whose
        // terminators move most visibly as the light flickers.
        auto bag = Mesh::create(IcosahedronGeometry::create(0.26f, 2),
                                matteMat(Color(0.22f, 0.19f, 0.15f), 0.85f));
        bag->position.set(-0.95f, 0.20f, -0.85f);
        bag->scale.set(1.f, 0.78f, 1.2f);
        g->add(bag);

        auto pot = Mesh::create(CylinderGeometry::create(0.14f, 0.11f, 0.20f, 20),
                                MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                        .color(Color(0.30f, 0.29f, 0.28f))
                                        .roughness(0.42f)
                                        .metalness(0.85f)));
        pot->position.set(0.62f, 0.10f, 0.62f);
        g->add(pot);
        return g;
    }

    std::shared_ptr<Group> makeGround() {
        auto g = Group::create();
        auto ground = Mesh::create(PlaneGeometry::create(60.f, 60.f, 1, 1),
                                   matteMat(Color(0.085f, 0.072f, 0.058f), 1.f));
        ground->rotateX(-math::PI / 2.f);
        g->add(ground);
        // Scattered rocks out to a few metres so the ground is not a featureless
        // plane where the light falls off — the falloff needs something to fall
        // off ON.
        for (int i = 0; i < 40; ++i) {
            const float a = h01(std::uint32_t(i), 31u) * kTau;
            const float r = 1.6f + 6.5f * h01(std::uint32_t(i), 37u);
            const float sc = 0.05f + 0.16f * h01(std::uint32_t(i), 41u);
            auto m = Mesh::create(IcosahedronGeometry::create(sc, 1),
                                  matteMat(Color(0.13f, 0.115f, 0.10f), 1.f));
            m->position.set(r * std::cos(a), sc * 0.4f, r * std::sin(a));
            m->rotation.set(h01(std::uint32_t(i), 43u) * kTau, h01(std::uint32_t(i), 47u) * kTau, 0.f);
            m->scale.set(1.f, 0.6f, 1.f);
            g->add(m);
        }
        return g;
    }

    // Night, pinned. Auto-exposure would swing on the fire (and on every
    // flicker of it), which is the documented capture confound in this repo:
    // it eats the fog and it eats the fire. Every capture and every A/B here
    // runs with a FIXED exposure.
    void setupNight(VulkanRenderer& renderer, Scene& scene) {
        renderer.toneMapping = ToneMapping::ACESFilmic;
        renderer.toneMappingExposure = 1.0f;
        renderer.setAutoExposure(false);
        renderer.setClearColor(Color(0.010f, 0.013f, 0.022f));
        scene.background = Color(0.010f, 0.013f, 0.022f);
        renderer.setDeferredStarfield(0.85f);
        renderer.setBloomIntensity(0.55f);
        renderer.setBloomThreshold(1.15f);
        // Ground mist. This is what turns the fire's PointLight into a visible
        // GLOW rather than a light that only exists where it lands on something
        // — the froxel injector scatters clustered lights through this medium,
        // and a low, shallow layer is where a campfire's glow actually sits.
        VulkanRenderer::HeightFogSettings hf;
        hf.density     = 0.032f;
        hf.baseY       = 0.0f;
        hf.falloff     = 2.4f;
        hf.noiseAmount = 0.55f;
        renderer.setHeightFog(hf);
        renderer.setVolumetricFog(true);
    }

}// namespace

int main(int argc, char** argv) {

    std::string shotPath;
    int   shotFrames = 260;
    float shotTime   = -1.f;// >= 0: freeze the effect at this absolute t
    bool  noEmbers   = false;
    bool  bench      = false;
    // TAA is what this ships and is gated on. The other two are here to LOOK
    // at, not to ship: a bright emissive volume is the worst case for the open
    // DLSS/FSR emitter-fog silhouette issue, which plan R-1 files under its own
    // bug rather than fighting here.
    std::string upscaler = "taa";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
        else if (a == "--t" && i + 1 < argc) shotTime = float(std::atof(argv[++i]));
        else if (a == "--no-embers") noEmbers = true;
        else if (a == "--bench") bench = true;
        else if (a == "--upscaler" && i + 1 < argc) upscaler = argv[++i];
    }
    const capture::Args cap = capture::parseArgs(argc, argv);

    Canvas canvas("Vulkan Deferred - Campfire",
                  {{"title", std::string("Vulkan Deferred - Campfire")},
                   {"size", std::pair<int, int>{1280, 720}},
                   {"vsync", false},
                   {"headless", !shotPath.empty() || bench}});
    VulkanRenderer renderer(canvas);
    renderer.setDlss(upscaler == "dlss");// F1 ships and gates on TAA; see plan R-1
    renderer.setFsr(upscaler == "fsr");

    Scene scene;
    setupNight(renderer, scene);

    scene.add(makeGround());
    scene.add(makeStoneRing(0.46f));
    scene.add(makeFuel());
    scene.add(makeCamp());

    // A moon: just enough to keep the sky-facing surfaces from being pure
    // black, so the firelight has something to be brighter THAN.
    auto moon = DirectionalLight::create(Color(0.55f, 0.66f, 0.95f), 0.05f);
    moon->position.set(-6.f, 9.f, -4.f);
    scene.add(moon);
    scene.add(AmbientLight::create(Color(0.20f, 0.26f, 0.40f), 0.05f));

    // ── The effect ──────────────────────────────────────────────────────────
    FireEffect::Params fp;
    fp.height = 1.05f;
    fp.radius = 0.25f;
    if (!noEmbers) {
        TextureLoader tl;
        fp.emberTexture = tl.load(std::string(DATA_FOLDER) + "/textures/smokeparticle.png",
                                  ColorSpace::sRGB);
    }
    fp.embers = !noEmbers && fp.emberTexture != nullptr;
    auto fire = FireEffect::create(fp);
    // Above the log bed, not on the ground plane: the flame's hottest, brightest
    // part is its base, and sinking that into the fuel hides the one thing the
    // emission ramp exists to show.
    fire->position.set(0.f, 0.13f, 0.f);
    scene.add(fire);
    fire->ignite();

    PerspectiveCamera camera(48.f, canvas.aspect(), 0.05f, 200.f);
    camera.position.set(1.55f, 1.05f, 2.75f);
    if (cap.camPos) camera.position.copy(*cap.camPos);
    const Vector3 target = cap.camTarget.value_or(Vector3(0.f, 0.62f, 0.f));
    camera.lookAt(target);

    std::unique_ptr<OrbitControls> controls;
    std::unique_ptr<RendererSettingsUi> ui;
    std::unique_ptr<KeyAdapter> keys;// the canvas keeps a reference, not a copy
    if (shotPath.empty() && !bench) {
        controls = std::make_unique<OrbitControls>(camera, canvas);
        controls->target.copy(target);
        controls->update();
        ui = std::make_unique<RendererSettingsUi>(canvas, renderer, [&] {
            ImGui::TextWrapped("One FireEffect: an emissive ParticleField for the flame, "
                               "a second field for the smoke, ember billboards, and one "
                               "PointLight. SPACE toggles it.");
        });
        keys = std::make_unique<KeyAdapter>(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
            if (evt.key == Key::SPACE) {
                if (fire->lit()) fire->extinguish();
                else fire->ignite();
            }
        });
        canvas.addKeyListener(*keys);
    }
    canvas.onWindowResize([&](WindowSize s) {
        camera.aspect = s.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(s);
    });

    // ── Bench: what does the whole effect cost on the GPU? ──────────────────
    // Interleaved A/B with the effect EXTINGUISHED as the A leg (the fields
    // stay in the scene, parked — so this measures the effect's work and not
    // the cost of two more entries), exposure pinned, vsync off, fixed dt.
    if (bench) {
        constexpr int kPairs = 6, kWarm = 40, kMeasure = 90;
        constexpr float kDt = 1.f / 60.f;
        struct Acc { double shade = 0, froxel = 0, density = 0, gpu = 0; };
        Acc onA{}, offA{};
        int frame = 0;
        const auto leg = [&](bool on, Acc& acc) {
            if (on) fire->ignite(); else fire->extinguish();
            for (int i = 0; i < kWarm; ++i) {
                fire->update(float(frame++) * kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
            }
            Acc a{};
            for (int i = 0; i < kMeasure; ++i) {
                fire->update(float(frame++) * kDt);
                canvas.animateOnce([&] { renderer.render(scene, camera); });
                const auto t = renderer.lastFrameTimings();
                a.shade   += t.pathTraceMs;
                a.froxel  += t.froxelMs;
                a.density += t.particleDensityMs;
                a.gpu     += t.gpuTotalMs;
            }
            acc.shade += a.shade / kMeasure;
            acc.froxel += a.froxel / kMeasure;
            acc.density += a.density / kMeasure;
            acc.gpu += a.gpu / kMeasure;
        };
        for (int p = 0; p < kPairs; ++p) {// A B A B ... , repo A/B rule
            leg(false, offA);
            leg(true, onA);
        }
        const double inv = 1.0 / double(kPairs);
        const auto sz = renderer.size();
        std::printf("\n[bench] %dx%d, %d interleaved pairs of %d frames, AE pinned, vsync off\n",
                    sz.width(), sz.height(), kPairs, kMeasure);
        std::printf("  %-22s %10s %10s %10s\n", "GPU ms/frame", "parked", "lit", "delta");
        const auto row = [&](const char* n, double off, double on) {
            std::printf("  %-22s %10.4f %10.4f %10.4f\n", n, off * inv, on * inv, (on - off) * inv);
        };
        row("deferred shade", offA.shade, onA.shade);
        row("froxel inj+int", offA.froxel, onA.froxel);
        row("density scatter", offA.density, onA.density);
        row("whole frame", offA.gpu, onA.gpu);
        std::printf("  effect total (shade + froxel + density delta): %.4f ms\n",
                    ((onA.shade - offA.shade) + (onA.froxel - offA.froxel) +
                     (onA.density - offA.density)) * inv);
        return 0;
    }

    // ── Headless capture ────────────────────────────────────────────────────
    if (!shotPath.empty()) {
        const int frames = cap.frames.value_or(shotFrames);
        constexpr float kDt = 1.f / 60.f;// fixed: no wall clock in a capture
        for (int i = 0; i < frames; ++i) {
            // The emitter is closed-form in t, so a capture can either run the
            // clock or freeze it. --t freezes: identical geometry every frame,
            // which is what lets TAA fully converge and makes two runs of this
            // command comparable byte for byte (bar the legacy embers).
            fire->update(shotTime >= 0.f ? shotTime : float(i) * kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }
        renderer.writeFramebuffer(shotPath);
        std::printf("[shot] %s (%d frames)\n", shotPath.c_str(), frames);
        if (cap.profile) capture::writeFrameTimings(renderer.lastFrameTimings(), cap, frames);
        return 0;
    }

    Clock clock;
    float t = 0.f;
    canvas.animate([&] {
        t += clock.getDelta();
        fire->update(t);
        renderer.render(scene, camera);
        if (ui) ui->render();
    });
    return 0;
}
