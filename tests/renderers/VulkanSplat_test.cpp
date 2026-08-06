// VulkanSplat_test — the three things the Gaussian-splat compute rasterizer
// has to be right about, checked on the GPU rather than argued about.
//
//   1. DETERMINISM. Two frames from the same camera must produce the same
//      sorted key array, the same payload array and the same composited
//      pixels. This is the gate on the tile EXPANSION: the obvious design —
//      every splat atomicAdds a global cursor and writes there — produces a
//      different (splat, tile) ordering every run, and the stable sort below
//      it then faithfully preserves that ordering into the image. The prefix
//      sum exists to make the ordering a pure function of the projection, and
//      this is how we know it does. Sensor goldens depend on it, and
//      "RT is never bit-exact" is already a scar in this tree.
//
//      The raster jitter is switched off (via setGbufferMsaa(2), which
//      rasterizes unjittered by design) so the two frames genuinely ARE the
//      same frame; with jitter on, the projection differs by a sub-pixel
//      offset and the keys legitimately differ.
//
//   2. OCCLUSION. The splat pass has no hardware depth test — it reads the
//      G-buffer depth and stops accumulating behind it. A box in front of half
//      the cloud must therefore hide exactly the half behind it, and the other
//      half must survive.
//
//   3. A GOLDEN IMAGE for a small procedural cloud — generated, not loaded, so
//      the repo needs no asset. Catches the whole pipeline drifting at once.
//
//      Byte-exact run to run in practice, but gated on PSNR/maxDelta rather
//      than equality: the temporal resolve's blend is frame-rate aware (see
//      taaDtFrames), so anything that shifts WHEN a frame lands — a
//      reallocation stall, a different machine — moves the last few frames'
//      convergence by a LSB or two. A golden captured under one splat-budget
//      constant and compared under another measured maxDelta 2 for exactly
//      that reason, which is inside the gate and should stay inside it.
//
// Run standalone (a plain exit-code program, not Catch2):
//   VulkanSplat_test              compare to tests/renderers/golden/splat.ppm
//   VulkanSplat_test --update     rewrite the reference after a reviewed change
// Exits 42 (CTest SKIP_RETURN_CODE) without a Vulkan GPU.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
namespace fs = std::filesystem;

namespace {

    constexpr int kW = 384, kH = 256;
    constexpr int kSkipCode = 42;
    // Enough frames for TAA to settle on a static camera; the golden is
    // compared after convergence, like VulkanGolden_test's.
    constexpr int kWarmFrames = 120;
    constexpr double kMinPsnr = 40.0;
    constexpr int kMaxDelta = 32;

    int failures = 0;
    void report(bool ok, const std::string& what) {
        std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

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
        f.get();
        std::vector<unsigned char> d(static_cast<size_t>(w) * h * 3);
        f.read(reinterpret_cast<char*>(d.data()), static_cast<std::streamsize>(d.size()));
        return f ? d : std::vector<unsigned char>{};
    }

    // A deterministic cloud with enough overlap that ordering matters and
    // enough opacity that the occlusion check has something to occlude.
    // SplatGenerator's xorshift is fixed-point by construction, so the same
    // seed gives the same cloud on every platform.
    SplatData makeCloud() {
        SplatGenerator::Options o;
        o.count = 6000;
        o.seed = 4242u;
        o.shDegree = 1;
        o.extent.set(3.f, 2.f, 3.f);
        o.minScale = 0.04f;
        o.maxScale = 0.11f;
        o.anisotropy = 3.f;
        o.minOpacity = 0.35f;
        o.maxOpacity = 0.95f;
        o.higherOrderAmplitude = 0.25f;
        auto data = SplatGenerator::generate(o);
        // Three hue bands by index, so a mis-sort is a colour error and not
        // only a subtle density one.
        for (size_t i = 0; i < data.count(); ++i) {
            const Vector3 hue = (i % 3 == 0)   ? Vector3{0.95f, 0.25f, 0.18f}
                                : (i % 3 == 1) ? Vector3{0.20f, 0.90f, 0.30f}
                                               : Vector3{0.22f, 0.35f, 0.95f};
            data.setDcColor(i, hue);
        }
        return data;
    }

    long long countLit(const std::vector<unsigned char>& rgb, int x0, int x1) {
        long long n = 0;
        for (int y = 0; y < kH; ++y)
            for (int x = x0; x < x1; ++x) {
                const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
                if (rgb[i] > 40 || rgb[i + 1] > 40 || rgb[i + 2] > 40) ++n;
            }
        return n;
    }

}// namespace

int main(int argc, char** argv) {
    bool update = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--update") == 0) update = true;

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanSplat_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.f;
    renderer.setRenderScale(1.f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setClearColor(Color(0.06f, 0.06f, 0.08f));
    // Unjittered raster (MSAA rasterizes without the Halton offset by design),
    // so "the same frame twice" really is the same frame.
    renderer.setGbufferMsaa(2);
    renderer.setSplatDebugChecksum(true);

    auto scene = Scene::create();
    scene->background = Color(0x0f0f14);

    auto camera = PerspectiveCamera::create(50, static_cast<float>(kW) / kH, 0.05f, 100.f);
    camera->position.set(0.f, 1.2f, 6.5f);
    camera->lookAt(Vector3{0.f, 0.f, 0.f});

    auto cloud = SplatCloud::create(makeCloud());
    scene->add(cloud);

    auto draw = [&] { renderer.render(*scene, *camera); };

    for (int i = 0; i < kWarmFrames; ++i) draw();

    // ── 1. determinism ──────────────────────────────────────────────────────
    std::uint64_t a[4]{}, b[4]{};
    draw();
    const bool gotA = renderer.splatDebugChecksum(a);
    draw();
    const bool gotB = renderer.splatDebugChecksum(b);

    report(gotA && gotB, "splat pass ran and reported a checksum");
    if (gotA && gotB) {
        std::printf("       keys %llu  payload %llu  pixels %llu  entries %llu\n",
                    static_cast<unsigned long long>(a[0]), static_cast<unsigned long long>(a[1]),
                    static_cast<unsigned long long>(a[2]), static_cast<unsigned long long>(a[3]));
        report(a[3] > 0, "the expansion produced (splat, tile) pairs");
        report(a[0] == b[0], "sorted KEY array is bit-identical across two frames");
        report(a[1] == b[1], "sorted PAYLOAD array is bit-identical across two frames");
        report(a[2] == b[2], "composited PIXELS are bit-identical across two frames");
        report(a[3] == b[3], "expanded entry COUNT is identical across two frames");
    }
    renderer.setSplatDebugChecksum(false);

    // ── 3. golden image (before the occluder joins the scene) ───────────────
    for (int i = 0; i < 8; ++i) draw();
    const auto pixels = renderer.readRGBPixels();
    const fs::path goldenPath = fs::path(PROJECT_FOLDER) / "tests" / "renderers" / "golden" / "splat.ppm";
    if (update) {
        writePPM(goldenPath, pixels, kW, kH);
        std::printf("[upd ] wrote %s\n", goldenPath.string().c_str());
    } else {
        int gw = 0, gh = 0;
        const auto ref = readPPM(goldenPath, gw, gh);
        if (ref.empty() || gw != kW || gh != kH) {
            report(false, "golden reference splat.ppm missing or wrong size (run --update)");
        } else {
            double mse = 0;
            int maxD = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const int d = std::abs(static_cast<int>(ref[i]) - static_cast<int>(pixels[i]));
                mse += static_cast<double>(d) * d;
                if (d > maxD) maxD = d;
            }
            mse /= static_cast<double>(ref.size());
            const double psnr = mse <= 0 ? 1e9 : 10.0 * std::log10(255.0 * 255.0 / mse);
            std::printf("       golden PSNR %.2f dB  maxDelta %d\n", psnr, maxD);
            report(psnr >= kMinPsnr && maxD <= kMaxDelta, "golden image matches the reference");
        }
    }

    // ── 2. splat-behind-mesh occlusion ──────────────────────────────────────
    // A slab across the LEFT half of the cloud, close to the camera. The left
    // half must lose most of its lit pixels; the right half must keep them.
    const auto before = pixels;
    auto box = Mesh::create(BoxGeometry::create(3.0f, 6.f, 0.2f),
                            MeshStandardMaterial::create({{"color", Color(0x101010)},
                                                          {"roughness", 1.f}}));
    box->position.set(-1.6f, 0.f, 2.6f);
    scene->add(box);
    for (int i = 0; i < 40; ++i) draw();
    const auto after = renderer.readRGBPixels();

    // The box covers roughly x < kW*0.42 at this framing; sample well inside
    // each side so the silhouette itself is not the measurement.
    const long long leftBefore  = countLit(before, 0, kW / 4);
    const long long leftAfter   = countLit(after, 0, kW / 4);
    const long long rightBefore = countLit(before, 3 * kW / 4, kW);
    const long long rightAfter  = countLit(after, 3 * kW / 4, kW);
    std::printf("       lit px  left %lld -> %lld   right %lld -> %lld\n",
                leftBefore, leftAfter, rightBefore, rightAfter);
    report(leftBefore > 500, "cloud covers the left side to begin with");
    report(leftAfter * 4 < leftBefore, "an opaque slab hides the splats behind it");
    report(rightAfter * 10 > rightBefore * 9, "splats beside the slab are untouched");

    // ── 3. delete, then load another — residency must follow the scene ──────
    // The residency cache is keyed by SplatCloud pointer. Two ways that goes
    // wrong, both exercised here: a deleted cloud's buffers staying resident
    // forever (a ~1.2 GB leak per 5M-splat import, and the OOM behind a
    // user-reported "crash when I deleted a splat and loaded a new one"), and a
    // NEW cloud allocated at the dead cloud's address being mistaken for it —
    // which renders the old scan under the new key with no error anywhere.
    scene->remove(*box);
    scene->remove(*cloud);
    const void* oldAddress = cloud.get();
    cloud.reset();// really freed — the recycling case needs the address back

    // Enough frames for lastSeen to age past framesInFlight_ + 1.
    for (int i = 0; i < 8; ++i) draw();
    report(renderer.splatResidentClouds() == 0,
           "a deleted cloud's GPU buffers are evicted once its frames drain");

    // Same allocation size, immediately after the free — on MSVC's allocator
    // this lands on the old address nearly always. If it does, the test bites
    // hardest; if not, it still covers plain delete-then-load.
    auto second = SplatCloud::create(makeCloud());
    std::printf("       address reuse: %s\n", second.get() == oldAddress ? "YES (recycled)" : "no");
    // A different transform: if stale residency served the OLD upload, the
    // hue-banded cloud would still draw at the ORIGIN rather than offset.
    second->position.set(2.0f, 0.f, 0.f);
    scene->add(second);
    for (int i = 0; i < 40; ++i) draw();
    const auto reloaded = renderer.readRGBPixels();
    report(renderer.splatResidentClouds() == 1, "the reloaded cloud is resident alone");
    // Offset +2 in x pushes the cloud right of centre: the right third must be
    // lit and the far-left third dark, which the ORIGIN-rendered stale version
    // fails on both counts.
    report(countLit(reloaded, 3 * kW / 4, kW) > 500,
           "and it draws at ITS transform, not the dead cloud's");
    report(countLit(reloaded, 0, kW / 6) < 200,
           "with nothing left where the dead cloud stood");

    // ── 4. hidden is parked, deleted is evicted, and the scratch follows ────
    // Toggling visibility must NOT cost the re-upload a real deletion does: a
    // hidden cloud stays in the scene, so its buffers stay on the device and
    // unhiding is instant. Only removal (or death) evicts — and when the last
    // cloud goes, the shared sort scratch (~700 MB at a 5M high-water) has to
    // go with it rather than linger as a high-water mark forever.
    second->visible = false;
    for (int i = 0; i < 10; ++i) draw();
    report(renderer.splatResidentClouds() == 1,
           "a HIDDEN cloud stays resident - parked, not evicted");
    const auto hidden = renderer.readRGBPixels();
    report(countLit(hidden, 3 * kW / 4, kW) < 200, "while drawing nothing");

    second->visible = true;
    for (int i = 0; i < 6; ++i) draw();
    const auto unhidden = renderer.readRGBPixels();
    report(countLit(unhidden, 3 * kW / 4, kW) > 500,
           "and unhiding draws again from the parked buffers");

    scene->remove(*second);
    second.reset();
    for (int i = 0; i < 10; ++i) draw();
    report(renderer.splatResidentClouds() == 0, "removal from the scene still evicts");
    report(renderer.splatScratchSplats() <= 1,
           "and the last eviction releases the shared sort scratch");

    std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
