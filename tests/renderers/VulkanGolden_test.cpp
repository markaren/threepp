// VulkanGolden_test — golden-image regression net for the Vulkan deferred path.
//
// The Vulkan deferred renderer + denoiser is the most regression-prone code in
// the tree (see the long history of temporal-artifact fixes: ghosting, slow-pan
// smear, fireflies, albedo demod, ReSTIR feedback loops, …) and had ZERO
// automated coverage. This renders a few fixed, deterministic scenes that
// exercise those paths and compares each to a committed reference, so a fix can
// no longer silently rot three changes later. Past the first three, each scene
// pins one stage of the deferred shade (sun shadows, the lamp list, probe GI,
// the fog froxels, clouds, the ocean branch, shadow maps, MSAA dispatch B), so
// a change that moves a stage shows up in the scene that owns it.
//
// Run standalone (it's a plain exit-code program, not Catch2):
//   VulkanGolden_test            compare to tests/renderers/golden/<name>.ppm;
//                                exit nonzero if any scene regresses
//   VulkanGolden_test --update   (re)write references — ONLY after an
//                                intentional, reviewed change to renderer output
// Or via CTest: `ctest -R VulkanGolden_test`. Exits 42 (→ CTest "Skipped") when
// no Vulkan/RT GPU is available, so CI without RT hardware doesn't fail.
//
// With THREEPP_VULKAN_NO_RT set (anything but empty or "0", the renderer's own
// rule) the renderer runs its raster path without ray queries, whose pictures
// differ from the traced ones by design, so those references live apart in
// tests/renderers/golden_nort/ under the same gates.
//
// References are PPM (raw readRGBPixels bytes — no codec/flip/channel ambiguity,
// byte-exact round-trip). They are GPU/driver-sensitive: the tolerance is set
// well above same-GPU run-to-run noise (measured ≈61 dB / maxD≤1), so regenerate
// with --update if you move to different hardware. This is a local pre-push
// check, not a hard cross-hardware CI gate.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "capture_util.hpp"// examples/libs (shared via target include dir)
#include "vulkan_test_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;
namespace fs = std::filesystem;

namespace {

    constexpr int kW = 384, kH = 256;
    constexpr int kFrames = 200;     // static camera → TAA/denoiser/accumulator converge
    constexpr double kMinPsnr = 40.0;// PSNR gate (measured same-GPU floor ≈ 61 dB)
    constexpr int kMaxDelta = 32;    // default per-channel gate — catches firefly/hot-pixel
                                     // regressions PSNR can miss (glass/metal run-to-run maxD ≤ 10).
                                     // Overridable per scene (see GoldenScene::maxDelta): the
                                     // emissive-only room drives ReSTIR DI at its noisiest, with a
                                     // measured run-to-run per-channel floor of ~70 on a few pixels
                                     // (hot% ~0.03), so a tight maxD flakes there. Its aggregate
                                     // PSNR stays ~51 dB and is the real regression signal.
    // Every frame of every scene renders at this one simulation time. The
    // shade's time inputs (the water's drift, the ocean spectrum, the cloud
    // clock) otherwise follow a clock that keeps running across the scenes, so
    // a scene's picture would depend on its place in the list; a still clock
    // also leaves the TAA blend at its one-frame weight, as the stepped
    // headless clock did.
    constexpr double kSimTime = 10.0;

    // Raw-RGB PPM I/O — what readRGBPixels gives us, byte-exact both ways.
    void writePPM(const fs::path& p, const std::vector<unsigned char>& rgb, int w, int h) {
        fs::create_directories(p.parent_path());
        std::ofstream f(p, std::ios::binary);
        f << "P6\n" << w << " " << h << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    }
    std::vector<unsigned char> readPPM(const fs::path& p, int& w, int& h) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return {};
        std::string magic;
        f >> magic;
        if (magic != "P6") return {};
        int maxv = 0;
        f >> w >> h >> maxv;
        f.get();// consume the single whitespace before the binary block
        std::vector<unsigned char> d(static_cast<size_t>(w) * h * 3);
        f.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(d.size()));
        if (!f) return {};
        return d;
    }

    struct GoldenScene {
        std::string name;
        std::function<void(Scene&, PerspectiveCamera&, const std::shared_ptr<Texture>&)> build;
        int maxDelta = kMaxDelta;// per-scene hot-pixel gate; default unless noisier
        // Renderer settings the scene needs beyond the harness's. Every scene
        // starts from the harness's settings, so one scene's switch never leaks
        // into the next.
        std::function<void(VulkanRenderer&)> setup;
    };

    std::shared_ptr<MeshStandardMaterial> matte(const Color& c, float roughness = 0.9f) {
        return MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(c).roughness(roughness).metalness(0.f));
    }

    std::shared_ptr<Mesh> groundPlane(float size, const Color& c) {
        auto g = Mesh::create(PlaneGeometry::create(size, size), matte(c));
        g->rotation.x = -math::PI / 2.f;
        return g;
    }

    // The emissive scene's closed matte room (inner faces, floor at y = -1),
    // dark outside and with no environment light.
    void addClosedRoom(Scene& s) {
        s.background = Color(0.01f, 0.01f, 0.015f);
        s.environment = nullptr;
        auto room = matte(Color(0.55f, 0.55f, 0.55f));
        room->side = Side::Back;
        auto box = Mesh::create(BoxGeometry::create(6.f, 4.f, 6.f), room);
        box->position.set(0.f, 1.f, 0.f);
        s.add(box);
    }

    // A spot light aimed at `at`. The target is a child of the scene so that it
    // lives as long as the light and its world matrix is kept current.
    void addSpot(Scene& s, const Color& c, float intensity, const Vector3& from, const Vector3& at,
                 float angle, float penumbra) {
        auto target = Object3D::create();
        target->position.copy(at);
        s.add(target);
        auto spot = SpotLight::create(c, intensity, 0.f, angle, penumbra, 2.f);
        spot->position.copy(from);
        spot->setTarget(*target);
        s.add(spot);
    }

    // The lamps scene: the closed room, its emissive cube, a glossy sphere to
    // carry the lamps' highlights, four coloured point lamps and one spot.
    void buildLampRoom(Scene& s, PerspectiveCamera& cam) {
        addClosedRoom(s);
        auto em = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color(1.f, 0.9f, 0.7f)).roughness(0.5f).metalness(0.f));
        em->emissive = Color(1.f, 0.85f, 0.55f);
        em->emissiveIntensity = 3.f;
        auto cube = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), em);
        cube->position.set(-1.2f, -0.7f, -1.2f);
        s.add(cube);
        auto glossy = Mesh::create(SphereGeometry::create(0.7f, 64, 32),
                                   MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                                        .color(Color(0.8f, 0.8f, 0.8f))
                                                                        .roughness(0.2f)
                                                                        .metalness(0.f)));
        glossy->position.set(0.3f, -0.3f, -0.2f);
        s.add(glossy);

        const struct {
            Color c;
            Vector3 p;
        } lamps[4] = {{Color(1.f, 0.25f, 0.2f), Vector3(-2.2f, 2.2f, -2.2f)},
                      {Color(0.2f, 1.f, 0.3f), Vector3(2.2f, 2.2f, -2.2f)},
                      {Color(0.25f, 0.4f, 1.f), Vector3(-2.2f, 0.2f, 1.6f)},
                      {Color(1.f, 0.8f, 0.3f), Vector3(1.8f, 0.0f, 0.4f)}};
        for (const auto& l : lamps) {
            auto p = PointLight::create(l.c, 2.f, 4.f, 2.f);
            p->position.copy(l.p);
            s.add(p);
        }
        // The spot sits in the far corner and points back toward the camera,
        // so the fog scene's beam scatters forward into the view.
        addSpot(s, Color(1.f, 0.95f, 0.85f), 80.f, Vector3(-2.6f, 2.8f, -2.6f), Vector3(0.9f, -1.f, 0.9f),
                0.3f, 0.25f);

        cam.position.set(2.0f, 1.5f, 2.4f);
        cam.lookAt(Vector3(-0.3f, 0.f, -0.6f));
    }

    // A sun over a ground plane with a box and a sphere casting onto it, under
    // the sky env. The sun is set soft so the penumbra is part of the picture.
    void buildSunYard(Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
        s.background = env;
        s.environment = env;
        s.add(groundPlane(30.f, Color(0.6f, 0.58f, 0.55f)));
        auto box = Mesh::create(BoxGeometry::create(1.f, 1.6f, 1.f), matte(Color(0.75f, 0.3f, 0.2f), 0.7f));
        box->position.set(-1.0f, 0.8f, 0.f);
        s.add(box);
        auto sph = Mesh::create(SphereGeometry::create(0.6f, 64, 32), matte(Color(0.2f, 0.4f, 0.75f), 0.5f));
        sph->position.set(1.1f, 0.9f, 0.6f);
        s.add(sph);
        auto sun = DirectionalLight::create(Color(1.f, 0.96f, 0.9f), 3.f);
        sun->position.set(-6.f, 7.f, -3.f);
        s.add(sun);
        cam.position.set(0.f, 3.2f, 6.f);
        cam.lookAt(Vector3(0.f, 0.3f, 0.f));
    }

    void sunShadowSetup(VulkanRenderer& r) {
        r.setSunAngularRadius(2.f);
    }

}// namespace

int main(int argc, char** argv) {
    bool update = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--update") == 0) update = true;
    }

    // Construction throws without a Vulkan/RT GPU (or a display) — treat that as
    // a CTest skip rather than a failure.
    auto gpu = vktest::openHeadlessVulkan("VulkanGolden_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(6.0f);
    renderer.setRenderScale(1.0f);// full-res readback, no upscale variance
    // Pin the golden gate to the plain TAA resolve — the one path every build
    // has. DLSS output shifts with every NGX model/DLL update (and across GPU
    // generations), and FSR is a compile-time option, so leaving either enabled
    // makes the references diverge between machines. Both are no-ops when the
    // feature isn't built in.
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    renderer.setClearColor(Color(0.f, 0.f, 0.f));

    // The harness's settings for everything a scene's setup may change, read
    // back rather than restated so the first three scenes keep rendering under
    // exactly the renderer's defaults.
    const uint32_t msaa0 = renderer.gbufferMsaa();
    const auto shadowMode0 = renderer.shadowMode();
    const bool probeGI0 = renderer.probeGI();
    const auto clouds0 = renderer.clouds();
    const auto heightFog0 = renderer.heightFog();
    const auto volumetrics0 = renderer.deferredVolumetrics();
    const float sunRadius0 = renderer.sunAngularRadius();
    auto resetSettings = [&] {
        renderer.setGbufferMsaa(msaa0);
        renderer.setShadowMode(shadowMode0);
        renderer.setProbeGI(probeGI0);
        renderer.setClouds(clouds0);
        renderer.setHeightFog(heightFog0);
        renderer.setDeferredVolumetrics(volumetrics0.first, volumetrics0.second);
        renderer.setSunAngularRadius(sunRadius0);
    };

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr");

    auto buildGlass = [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
        s.background = env;
        s.environment = env;
        const Color cols[3] = {Color(0.85f, 0.12f, 0.10f), Color(0.10f, 0.75f, 0.18f),
                               Color(0.12f, 0.28f, 0.88f)};
        for (int i = 0; i < 3; ++i) {
            auto m = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(cols[i]).roughness(0.6f).metalness(0.f));
            auto b = Mesh::create(BoxGeometry::create(1.1f, 1.1f, 0.3f), m);
            b->position.set((i - 1) * 1.4f, 0.f, -2.f);
            s.add(b);
        }
        auto glass = MeshPhysicalMaterial::create();
        glass->color = Color::white;
        glass->roughness = 0.04f;
        glass->metalness = 0.f;
        glass->transmission = 1.f;
        glass->setIor(1.5f);
        glass->thickness = 1.0f;
        auto sph = Mesh::create(SphereGeometry::create(1.0f, 64, 32), glass);
        s.add(sph);
        cam.position.set(0.f, 0.5f, 3.6f);
        cam.lookAt(Vector3(0.f, 0.f, -1.f));
    };

    auto addMetalSpheres = [](Scene& s) {
        const float rough[2] = {0.08f, 0.35f};
        for (int i = 0; i < 2; ++i) {
            auto m = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                          .color(Color(0.95f, 0.93f, 0.88f))
                                                          .metalness(1.f)
                                                          .roughness(rough[i]));
            auto sp = Mesh::create(SphereGeometry::create(0.9f, 64, 32), m);
            sp->position.set((i == 0 ? -1.1f : 1.1f), 0.f, 0.f);
            s.add(sp);
        }
    };

    // The scenes render one after another on one renderer, and what a scene
    // leaves in it (the temporal histories, the environment it bound) reaches
    // the next, so a reference holds for this list in this order: a new scene
    // goes at the end. The floors in the comments are two runs of the same
    // binary against references it wrote, on one GPU, with and without RT.
    const std::vector<GoldenScene> scenes = {
            // Glass sphere over a colour triptych: refraction through a closed
            // dielectric + env reflection. The transmission/ordering path.
            {"glass", buildGlass},
            // Two metal spheres (smooth + glossy) under IBL: specular reflection,
            // Fresnel, and the GGX multiscatter energy-compensation path.
            {"metal", [&](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
                 s.background = env;
                 s.environment = env;
                 addMetalSpheres(s);
                 cam.position.set(0.f, 0.4f, 3.6f);
                 cam.lookAt(Vector3(0.f, 0.f, 0.f));
             }},
            // Emissive cube as the ONLY light in a matte room (env disabled):
            // the emissive-NEE / direct-lighting determinism repro.
            {"emissive", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>&) {
                 addClosedRoom(s);
                 auto em = MeshStandardMaterial::create(
                         MeshStandardMaterial::Params{}.color(Color(1.f, 0.9f, 0.7f)).roughness(0.5f).metalness(0.f));
                 em->emissive = Color(1.f, 0.85f, 0.55f);
                 em->emissiveIntensity = 6.f;
                 auto cube = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), em);
                 cube->position.set(0.f, 0.6f, 0.f);
                 s.add(cube);
                 cam.position.set(2.0f, 1.5f, 2.4f);
                 cam.lookAt(Vector3(0.f, 0.6f, 0.f));
             },
             /*maxDelta=*/96},// ReSTIR DI floor ~70 here; gate above it (PSNR is the real signal)
            // A soft sun over a yard: the sun's shadow rays, the angular-radius
            // jitter and the shadow-ratio channel that denoises them. Floor:
            // bit-exact with and without RT.
            {"sun_shadows", buildSunYard, kMaxDelta, sunShadowSetup},
            // The same yard with the sun's shadows from cascaded maps instead of
            // rays: the shadow-map permutation of the shade. Without RT this is
            // the same picture as sun_shadows, whose Auto mode runs maps there.
            // Floor: bit-exact with and without RT.
            {"shadow_maps", buildSunYard, kMaxDelta, [](VulkanRenderer& r) {
                 sunShadowSetup(r);
                 r.setShadowMode(VulkanRenderer::ShadowMode::ShadowMaps);
             }},
            // Four coloured point lamps and a spot in the closed room: the
            // clustered light list, ReSTIR DI choosing among several lamps and
            // the lamps' specular on a glossy sphere. Floor: bit-exact with and
            // without RT. The gate is emissive's all the same: a perturbation
            // upstream of the reservoirs (here, a run without an environment
            // bound before it) moved the picture to 43 dB / maxD 35.
            {"lamps", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>&) {
                 buildLampRoom(s, cam);
             },
             /*maxDelta=*/96},
            // The lamp room filled with height fog and volumetric spot beams:
            // the froxel medium, the fog legs on direct light, the GI
            // recombine and the mirror ray through the fog. The spot faces the
            // camera so its beam scatters forward into the view. Floor:
            // bit-exact with and without RT; gated as lamps for the same reason.
            {"fog_froxels", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>&) {
                 buildLampRoom(s, cam);
             },
             /*maxDelta=*/96, [](VulkanRenderer& r) {
                 VulkanRenderer::HeightFogSettings hf;
                 hf.density = 0.2f;
                 hf.baseY = -1.f;
                 hf.falloff = 6.f;
                 hf.noiseAmount = 0.f;// a smooth profile, independent of the noise field
                 r.setHeightFog(hf);
                 r.setDeferredVolumetrics(0.3f, 0.6f);
             }},
            // A room open to the sky on one side and lit only through that
            // opening: the GI gather falls back to the probe grid for the
            // bounces the one-bounce gather cannot reach. Without RT there is no
            // probe grid (setProbeGI(true) is refused) and the room shows the
            // cosmetic ambient instead. Floor: bit-exact with and without RT.
            {"probe_gi", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
                 s.background = env;
                 s.environment = env;
                 auto wallMat = matte(Color(0.6f, 0.58f, 0.55f));
                 wallMat->side = Side::Double;
                 auto plane = [&](float w, float h, const Vector3& p, const Euler& rot) {
                     auto m = Mesh::create(PlaneGeometry::create(w, h), wallMat);
                     m->position.copy(p);
                     m->rotation.copy(rot);
                     s.add(m);
                 };
                 // Floor, ceiling, back and side walls; the +X wall is left out.
                 plane(6.f, 6.f, Vector3(0.f, -1.f, 0.f), Euler(-math::PI / 2.f, 0.f, 0.f));
                 plane(6.f, 6.f, Vector3(0.f, 3.f, 0.f), Euler(math::PI / 2.f, 0.f, 0.f));
                 plane(6.f, 4.f, Vector3(0.f, 1.f, -3.f), Euler(0.f, 0.f, 0.f));
                 plane(6.f, 4.f, Vector3(0.f, 1.f, 3.f), Euler(0.f, math::PI, 0.f));
                 plane(6.f, 4.f, Vector3(-3.f, 1.f, 0.f), Euler(0.f, math::PI / 2.f, 0.f));
                 auto block = Mesh::create(BoxGeometry::create(1.f, 1.2f, 1.f), matte(Color(0.7f, 0.35f, 0.25f)));
                 block->position.set(-1.2f, -0.4f, -0.8f);
                 s.add(block);
                 cam.position.set(-2.6f, 1.0f, 2.7f);
                 cam.lookAt(Vector3(1.0f, 0.7f, -1.5f));
             },
             kMaxDelta, [](VulkanRenderer& r) { r.setProbeGI(true); }},
            // The metal spheres on a ground plane under a cloud deck with the
            // sun low: the cloud march over the sky and its composite over the
            // ground and in the reflections. Wind and churn are stopped so the
            // deck holds still. Floor: bit-exact with and without RT.
            {"clouds", [&](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
                 s.background = env;
                 s.environment = env;
                 addMetalSpheres(s);
                 auto ground = groundPlane(400.f, Color(0.35f, 0.33f, 0.3f));
                 ground->position.y = -0.9f;
                 s.add(ground);
                 auto sun = DirectionalLight::create(Color(1.f, 0.85f, 0.65f), 3.f);
                 sun->position.set(-20.f, 4.f, -30.f);
                 s.add(sun);
                 cam.position.set(0.f, 0.2f, 4.2f);
                 cam.lookAt(Vector3(0.f, 1.2f, 0.f));
             },
             kMaxDelta, [](VulkanRenderer& r) {
                 VulkanRenderer::CloudSettings cs;
                 cs.coverage = 0.55f;
                 cs.bottomY = 600.f;
                 cs.topY = 1400.f;
                 cs.wind = Vector3(0.f, 0.f, 0.f);
                 cs.evolveSpeed = 0.f;
                 r.setClouds(cs);
             }},
            // An FFT ocean with a box floating in it under the sky env: the
            // water branch of the shade, its reflection and the fine-cascade
            // normal, with the wake samplers at their dummies. The one scene
            // that is not bit-exact run to run: floor 80 dB / maxD 2 with RT,
            // 92 dB / maxD 1 without.
            {"ocean", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
                 s.background = env;
                 s.environment = env;
                 Ocean::Options opts;
                 opts.size = 400.f;
                 s.add(Ocean::create(opts));
                 auto box = Mesh::create(BoxGeometry::create(2.f, 1.f, 2.f), matte(Color(0.8f, 0.45f, 0.15f), 0.6f));
                 box->position.set(0.f, 0.1f, 0.f);
                 s.add(box);
                 cam.position.set(0.f, 4.f, 10.f);
                 cam.lookAt(Vector3(0.f, 0.f, 0.f));
             }},
            // The glass scene with a 4x G-buffer: the per-sample edge shading of
            // dispatch B, its coverage accounting and per-sample ray origins.
            // Floor: bit-exact with and without RT.
            {"msaa4_glass", buildGlass, kMaxDelta, [](VulkanRenderer& r) { r.setGbufferMsaa(4); }},
    };

    // The renderer's own reading of the variable (VulkanContext): set and not
    // starting with '0' turns ray tracing off.
    const char* noRtEnv = std::getenv("THREEPP_VULKAN_NO_RT");
    const bool noRt = noRtEnv && *noRtEnv && *noRtEnv != '0';
    const fs::path goldenDir = fs::path(PROJECT_FOLDER) / "tests" / "renderers" / (noRt ? "golden_nort" : "golden");

    size_t sceneIdx = 0;
    int frame = 0, failures = 0, missing = 0;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<PerspectiveCamera> camera;

    auto buildCurrent = [&] {
        resetSettings();
        if (scenes[sceneIdx].setup) scenes[sceneIdx].setup(renderer);
        scene = std::make_unique<Scene>();
        camera = std::make_unique<PerspectiveCamera>(45.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
        scenes[sceneIdx].build(*scene, *camera, env);
        camera->updateMatrixWorld();
    };

    auto finish = [&] {
        if (update)
            std::printf("updated %zu references in %s\n", scenes.size(), goldenDir.string().c_str());
        else
            std::printf("golden: %d/%zu failed, %d missing (gate: PSNR>=%.0f dB, maxD per-scene)\n",
                        failures, scenes.size(), missing, kMinPsnr);
        std::exit((update || (failures == 0 && missing == 0)) ? 0 : 1);
    };

    buildCurrent();// first scene

    // Render through canvas.animate — the proven Vulkan present path (the --shot
    // capture loop uses the same). A static camera over kFrames lets the
    // accumulator / denoiser / TAA converge before we read the pixels.
    canvas.animate([&] {
        renderer.setSimTime(kSimTime);
        renderer.render(*scene, *camera);
        if (++frame < kFrames) return;

        const std::vector<unsigned char> px = renderer.readRGBPixels();
        const auto& gs = scenes[sceneIdx];
        const fs::path ref = goldenDir / (gs.name + ".ppm");
        if (update) {
            writePPM(ref, px, kW, kH);
            std::printf("[update] wrote %s (%dx%d)\n", ref.string().c_str(), kW, kH);
        } else {
            int rw = 0, rh = 0;
            const std::vector<unsigned char> golden = readPPM(ref, rw, rh);
            if (golden.empty() || rw != kW || rh != kH || golden.size() != px.size()) {
                std::printf("[%s] MISSING/!match reference %s - run --update\n",
                            gs.name.c_str(), ref.string().c_str());
                ++missing;
            } else {
                const capture::DiffResult d = capture::imageDiff(px, golden);
                const bool pass = d.psnr >= kMinPsnr && d.maxD <= gs.maxDelta;
                std::printf("[%s] PSNR=%5.1f dB  maxD=%3d (<=%d)  hot=%6.3f%%  ->  %s\n",
                            gs.name.c_str(), d.psnr, d.maxD, gs.maxDelta, d.hotPct, pass ? "PASS" : "FAIL");
                if (!pass) ++failures;
            }
        }

        frame = 0;
        if (++sceneIdx >= scenes.size()) finish();// prints summary + exits
        buildCurrent();
    });
    return 0;
}
