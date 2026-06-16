// VulkanGolden_test — golden-image regression net for the Vulkan PT/deferred path.
//
// The Vulkan path tracer + denoiser is the most regression-prone code in the
// tree (see the long history of temporal-artifact fixes: ghosting, slow-pan
// smear, fireflies, albedo demod, ReSTIR feedback loops, …) and had ZERO
// automated coverage. This renders a few fixed, deterministic scenes that
// exercise those paths and compares each to a committed reference, so a fix can
// no longer silently rot three changes later.
//
// Run standalone (it's a plain exit-code program, not Catch2):
//   VulkanGolden_test            compare to tests/renderers/golden/<name>.ppm;
//                                exit nonzero if any scene regresses
//   VulkanGolden_test --update   (re)write references — ONLY after an
//                                intentional, reviewed change to renderer output
//   VulkanGolden_test --pt       use the ReferencePT path (<name>_pt.ppm)
// Or via CTest: `ctest -R VulkanGolden_test`. Exits 42 (→ CTest "Skipped") when
// no Vulkan/RT GPU is available, so CI without RT hardware doesn't fail.
//
// References are PPM (raw readRGBPixels bytes — no codec/flip/channel ambiguity,
// byte-exact round-trip). They are GPU/driver-sensitive: the tolerance is set
// well above same-GPU run-to-run noise (measured ≈61 dB / maxD≤1), so regenerate
// with --update if you move to different hardware. This is a local pre-push
// check, not a hard cross-hardware CI gate.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "capture_util.hpp"// examples/vulkan (shared via target include dir)

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    constexpr int kW = 384, kH = 256;
    constexpr int kFrames = 200;     // static camera → TAA/denoiser/accumulator converge
    constexpr double kMinPsnr = 40.0;// PSNR gate (measured same-GPU floor ≈ 61 dB)
    constexpr int kMaxDelta = 32;    // per-channel gate — catches firefly/hot-pixel
                                     // regressions PSNR can miss (run-to-run maxD ≤ 1)
    constexpr int kSkipCode = 42;    // CTest SKIP_RETURN_CODE (no Vulkan/RT GPU)

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
    };

}// namespace

int main(int argc, char** argv) {
    bool update = false, usePT = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--update") == 0) update = true;
        else if (std::strcmp(argv[i], "--pt") == 0) usePT = true;
    }

    // Construction throws without a Vulkan/RT GPU (or a display) — treat that as
    // a CTest skip rather than a failure.
    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanGolden_test").size(kW, kH).vsync(false));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    renderer.setDenoise(true);
    renderer.setRestirDIEnabled(true);
    renderer.setFireflyClamp(6.0f);
    renderer.setMaxBounces(4);
    renderer.setRenderScale(1.0f);// full-res readback, no upscale variance
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    renderer.setClearColor(Color(0.f, 0.f, 0.f));
    if (usePT) renderer.setRenderMode(VulkanRenderer::RenderMode::ReferencePT);

    RGBELoader rgbe;
    auto env = rgbe.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr");

    const std::vector<GoldenScene> scenes = {
            // Glass sphere over a colour triptych: refraction through a closed
            // dielectric + env reflection. The transmission/ordering path.
            {"glass", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
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
             }},
            // Two metal spheres (smooth + glossy) under IBL: specular reflection,
            // Fresnel, and the GGX multiscatter energy-compensation path.
            {"metal", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>& env) {
                 s.background = env;
                 s.environment = env;
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
                 cam.position.set(0.f, 0.4f, 3.6f);
                 cam.lookAt(Vector3(0.f, 0.f, 0.f));
             }},
            // Emissive cube as the ONLY light in a matte room (env disabled):
            // the emissive-NEE / direct-lighting determinism repro.
            {"emissive", [](Scene& s, PerspectiveCamera& cam, const std::shared_ptr<Texture>&) {
                 s.background = Color(0.01f, 0.01f, 0.015f);
                 s.environment = nullptr;// no IBL — the cube must light the room
                 auto room = MeshStandardMaterial::create(
                         MeshStandardMaterial::Params{}.color(Color(0.55f, 0.55f, 0.55f)).roughness(0.9f).metalness(0.f));
                 room->side = Side::Back;// render the inner faces; camera sits inside
                 auto box = Mesh::create(BoxGeometry::create(6.f, 4.f, 6.f), room);
                 box->position.set(0.f, 1.f, 0.f);
                 s.add(box);
                 auto em = MeshStandardMaterial::create(
                         MeshStandardMaterial::Params{}.color(Color(1.f, 0.9f, 0.7f)).roughness(0.5f).metalness(0.f));
                 em->emissive = Color(1.f, 0.85f, 0.55f);
                 em->emissiveIntensity = 6.f;
                 auto cube = Mesh::create(BoxGeometry::create(0.6f, 0.6f, 0.6f), em);
                 cube->position.set(0.f, 0.6f, 0.f);
                 s.add(cube);
                 cam.position.set(2.0f, 1.5f, 2.4f);
                 cam.lookAt(Vector3(0.f, 0.6f, 0.f));
             }},
    };

    const fs::path goldenDir = fs::path(PROJECT_FOLDER) / "tests" / "renderers" / "golden";
    const std::string suffix = usePT ? "_pt" : "";

    size_t sceneIdx = 0;
    int frame = 0, failures = 0, missing = 0;
    std::unique_ptr<Scene> scene;
    std::unique_ptr<PerspectiveCamera> camera;

    auto buildCurrent = [&] {
        scene = std::make_unique<Scene>();
        camera = std::make_unique<PerspectiveCamera>(45.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
        scenes[sceneIdx].build(*scene, *camera, env);
        camera->updateMatrixWorld();
    };

    auto finish = [&] {
        if (update)
            std::printf("updated %zu references in %s\n", scenes.size(), goldenDir.string().c_str());
        else
            std::printf("golden: %d/%zu failed, %d missing (gate: PSNR>=%.0f dB, maxD<=%d)\n",
                        failures, scenes.size(), missing, kMinPsnr, kMaxDelta);
        std::exit((update || (failures == 0 && missing == 0)) ? 0 : 1);
    };

    buildCurrent();// first scene; renderer isn't built yet, so NO resetAccumulation here

    // Render through canvas.animate — the proven Vulkan present path (the --shot
    // capture loop uses the same). A static camera over kFrames lets the
    // accumulator / denoiser / TAA converge before we read the pixels.
    canvas.animate([&] {
        renderer.render(*scene, *camera);
        if (++frame < kFrames) return;

        const std::vector<unsigned char> px = renderer.readRGBPixels();
        const auto& gs = scenes[sceneIdx];
        const fs::path ref = goldenDir / (gs.name + suffix + ".ppm");
        if (update) {
            writePPM(ref, px, kW, kH);
            std::printf("[update] wrote %s (%dx%d)\n", ref.string().c_str(), kW, kH);
        } else {
            int rw = 0, rh = 0;
            const std::vector<unsigned char> golden = readPPM(ref, rw, rh);
            if (golden.empty() || rw != kW || rh != kH || golden.size() != px.size()) {
                std::printf("[%s] MISSING/!match reference %s — run --update\n",
                            gs.name.c_str(), ref.string().c_str());
                ++missing;
            } else {
                const capture::DiffResult d = capture::imageDiff(px, golden);
                const bool pass = d.psnr >= kMinPsnr && d.maxD <= kMaxDelta;
                std::printf("[%s] PSNR=%5.1f dB  maxD=%3d  hot=%6.3f%%  ->  %s\n",
                            gs.name.c_str(), d.psnr, d.maxD, d.hotPct, pass ? "PASS" : "FAIL");
                if (!pass) ++failures;
            }
        }

        frame = 0;
        if (++sceneIdx >= scenes.size()) finish();// prints summary + exits
        renderer.resetAccumulation();// next scene starts clean (renderer is built now)
        buildCurrent();
    });
    return 0;
}
