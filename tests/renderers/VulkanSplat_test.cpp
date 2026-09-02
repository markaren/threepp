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

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    // SplatGenerator draws from math::Rng (fixed-point by construction), so
    // the same seed gives the same cloud on every platform.
    //
    // The x-extent is wider than the visible middle band ON PURPOSE: the
    // outer-quarter AOV checks below need solidly-owned pixels beside the
    // slab, and at 3.0 the outer quarters were covered only by whichever
    // stretched fringe splats the seed happened to deal — a fixture that
    // breaks on any RNG change without anything being wrong.
    SplatData makeCloud() {
        SplatGenerator::Options o;
        o.count = 6000;
        o.seed = 4242u;
        o.shDegree = 1;
        o.extent.set(4.5f, 2.f, 3.f);
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

    // ── 3b. the reflection volume: baked, sized, non-empty ──────────────────
    // Each resident cloud is voxelized ONCE at upload into an rgba16f medium
    // (rgb = linear radiance, a = sigma_t per local metre) that reflection legs
    // march — the splat counterpart of the fire-in-the-mirror density volume,
    // and the reason a cloud BEHIND the camera can show up in water. Nothing on
    // the primary leg reads it, so none of the goldens above can move.
    std::uint64_t vol1[3]{};
    renderer.splatVolumeHash(vol1);
    const std::uint64_t volBytes1 = renderer.splatVolumeBytes();
    const std::uint64_t volGen1   = renderer.splatVolumeGeneration();
    std::printf("       volume  hash %llu  texels %llu  occupied %llu  bytes %llu  gen %llu\n",
                static_cast<unsigned long long>(vol1[0]), static_cast<unsigned long long>(vol1[1]),
                static_cast<unsigned long long>(vol1[2]),
                static_cast<unsigned long long>(volBytes1),
                static_cast<unsigned long long>(volGen1));
    report(vol1[1] > 0, "the cloud was baked into a reflection volume");
    // rgba16f is 8 bytes a texel, and volumeBytes() is the assertable form of
    // "the volume is resident" the way residentCount() is for the buffers.
    report(volBytes1 == vol1[1] * 8, "and volumeBytes() agrees with the texel count");
    // A bake that deposited NOTHING would be reproducibly empty and would sail
    // through the determinism check below, so the occupancy is asserted first.
    report(vol1[2] > 1000, "with splats actually deposited into it");
    report(volGen1 > 0, "and the bake bumped the volume generation");

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

    // ── 2b. the expected-depth AOV ──────────────────────────────────────────
    // The raster accumulates an alpha-weighted expected view distance to make
    // the cloud's motion vectors computable; setSplatDepthAov exports it. It is
    // the only record a frame keeps that a splat cloud was THERE — the cloud is
    // in no acceleration structure, so nothing ray-traced can find it.
    //
    // Checked here rather than in its own program because the scene is already
    // the useful one: a cloud at a known distance with a slab across half of
    // it, so the AOV can be asserted to agree with the software depth test
    // rather than merely to be non-empty.
    {
        std::vector<uint8_t> aov;
        int aw = 0, ah = 0, abpp = 0;
        report(!renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                        aov, aw, ah, abpp),
               "the depth AOV reads back FALSE until it is asked for");

        renderer.setSplatDepthAov(true);
        report(renderer.splatDepthAov(), "setSplatDepthAov(true) sticks");
        // The toggle reallocates the render targets, so give the frame that
        // rebuilds them a chance to draw before reading.
        for (int i = 0; i < 4; ++i) draw();

        const bool got = renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                                 aov, aw, ah, abpp);
        std::printf("       AOV extent %dx%d  bpp %d (display %dx%d)\n",
                    aw, ah, abpp, kW, kH);
        // The G-buffer extent, NOT the display extent: readGBufferAOV documents
        // that it hands back the render-scaled size, and an upscaler makes the
        // two differ. Asserting kW/kH here would be asserting that no upscaler
        // is configured, which is a different test.
        const bool shaped = got && abpp == 4 && aw > 0 && ah > 0 &&
                            aov.size() == static_cast<size_t>(aw) * ah * 4;
        report(shaped, "the depth AOV reads back at the G-buffer extent as 1x float32");

        if (shaped) {
            std::vector<float> d(static_cast<size_t>(aw) * ah);
            std::memcpy(d.data(), aov.data(), d.size() * sizeof(float));

            // Distances only where a cloud owns the pixel; 0 is the sentinel.
            long long covered = 0;
            double sum = 0;
            float lo = 1e30f, hi = -1e30f;
            bool anyNegative = false;
            for (float v : d) {
                if (v < 0.f) anyNegative = true;
                if (v <= 0.f) continue;
                ++covered;
                sum += v;
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            std::printf("       AOV covered %lld px  dist [%.2f, %.2f]  mean %.2f\n",
                        covered, covered ? lo : 0.f, covered ? hi : 0.f,
                        covered ? sum / static_cast<double>(covered) : 0.0);

            report(covered > 2000, "the AOV marks the pixels the cloud owns");
            report(!anyNegative, "and never a negative distance");
            // Camera sits 6.6 units from the origin; the cloud's own extent is
            // 3x2x3, so every covered pixel has to fall inside a generous
            // bracket around that. A reversed-Z NDC value (0..1) or a
            // world-space coordinate would both fail this.
            report(covered > 0 && lo > 3.f && hi < 12.f,
                   "at view-space distances consistent with where the cloud is");

            // The AOV honours the same software depth test the colour does:
            // behind the slab the accumulation stops, so those pixels stay 0.
            auto countCovered = [&](int x0, int x1) {
                long long n = 0;
                for (int y = 0; y < ah; ++y)
                    for (int x = x0; x < x1; ++x)
                        if (d[static_cast<size_t>(y) * aw + x] > 0.f) ++n;
                return n;
            };
            const long long leftCov  = countCovered(0, aw / 4);
            const long long rightCov = countCovered(3 * aw / 4, aw);
            std::printf("       AOV covered  left %lld   right %lld\n", leftCov, rightCov);
            // Far fewer than countLit reports for the same strip, and that is
            // the coverage gate doing its job: a pixel the cloud merely tints
            // is lit, but only a pixel the cloud actually OWNS (accumulated
            // alpha past 0.5) gets a depth. The outer quarter is mostly thin
            // fringe, so the two counts differ by roughly 7x here.
            report(rightCov > 50, "the AOV is populated beside the slab");
            report(leftCov * 4 < rightCov,
                   "and the slab occludes the AOV as it occludes the colour");

            // ── 2c. the MEDIAN statistic (plans/splat-surface-bake.md P0) ───
            // Expected depth is the cloud's opacity centroid along the ray, so
            // it sits BEHIND the visible front by roughly the cloud's own
            // thickness; the median — the transmittance-0.5 crossing — is the
            // surface estimate depth fusion needs. The bias between them is
            // one-signed, which is the part worth asserting rather than the
            // magnitude, which is the cloud's thickness and nothing else.
            const std::vector<float> dExp = d;
            const std::vector<uint8_t> expBytes = aov;

            // The same frame twice: the raster is unjittered here (MSAA 2), so
            // "deterministic" means bit-identical, not close.
            std::vector<uint8_t> again;
            int rw = 0, rh = 0, rb = 0;
            draw();
            const bool detExpected =
                    renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                            again, rw, rh, rb) &&
                    again == expBytes;

            renderer.setSplatDepthAov(VulkanRenderer::SplatDepthMode::Median);
            for (int i = 0; i < 4; ++i) draw();
            std::vector<uint8_t> med1, med2;
            const bool gotMed = renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                                        med1, rw, rh, rb);
            draw();
            const bool detMedian =
                    gotMed && med1.size() == expBytes.size() &&
                    renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                            med2, rw, rh, rb) &&
                    med2 == med1;

            if (gotMed && med1.size() == expBytes.size()) {
                std::vector<float> dMed(dExp.size());
                std::memcpy(dMed.data(), med1.data(), med1.size());

                std::vector<float> diff;
                diff.reserve(dExp.size());
                long long inFront = 0;
                double sumDiff = 0;
                for (size_t i = 0; i < dExp.size(); ++i) {
                    if (!(dExp[i] > 0.f) || !(dMed[i] > 0.f)) continue;
                    diff.push_back(dExp[i] - dMed[i]);
                    sumDiff += diff.back();
                    if (diff.back() > 1e-4f) ++inFront;
                }
                const long long both = static_cast<long long>(diff.size());
                std::sort(diff.begin(), diff.end());
                const double meanDiff = both ? sumDiff / static_cast<double>(both) : 0.0;
                const double midDiff  = both ? diff[diff.size() / 2] : 0.0;
                const double frontFrac = both ? static_cast<double>(inFront) / static_cast<double>(both) : 0.0;
                std::printf("       expected - median: %lld px  typical %+.4f  mean %+.4f"
                            "  range [%+.4f, %+.4f]  in front %.1f%%\n",
                            both, midDiff, meanDiff, both ? diff.front() : 0.f,
                            both ? diff.back() : 0.f, 100.0 * frontFrac);
                // The TYPICAL difference, not the mean: as coverage approaches
                // the 0.5 gate from above, the transmittance-0.5 crossing
                // degenerates to the LAST contributing splat and lands behind
                // the expected value — a real property of the statistic, not a
                // bug, and the reason the mean over covered pixels is ~0 while
                // four pixels in five have the median in front. A fusion
                // consumer weights by coverage; this test only asserts the
                // sign of the bias where the cloud is solid.
                report(both > 2000 && midDiff > 0.005 && frontFrac > 0.7,
                       "the median depth sits in front of the expected depth");
            } else {
                report(false, "the median-mode AOV reads back at the same extent");
            }
            report(detExpected && detMedian,
                   "both AOV modes are bit-identical across two renders");
            renderer.setSplatDepthAov(VulkanRenderer::SplatDepthMode::Expected);
        }
    }

    // ── 2d. point mode ──────────────────────────────────────────────────────
    // SplatCloud::setPointMix(1): every splat an opaque 2 px disc, through the
    // same sort, the same software depth test and the same compositor. The
    // slab is still in the scene, so the assertions are the occlusion ones
    // again, on dots — plus that dots cover fewer pixels than the Gaussians
    // they replace, that the frame is still deterministic, and that mix 0
    // brings the Gaussian frame back.
    {
        cloud->setPointMix(1.f);
        cloud->setPointSize(2.f);
        for (int i = 0; i < 40; ++i) draw();
        const auto dots = renderer.readRGBPixels();
        const long long leftDots  = countLit(dots, 0, kW / 4);
        const long long rightDots = countLit(dots, 3 * kW / 4, kW);
        std::printf("       point mode lit px  left %lld  right %lld (gaussians %lld)\n",
                    leftDots, rightDots, rightAfter);
        report(rightDots > 300, "point mode draws the cloud as dots");
        report(rightDots < rightAfter, "2 px discs cover fewer pixels than the Gaussians did");
        report(leftDots * 4 < leftBefore, "the opaque slab hides the points behind it");

        std::uint64_t pa[4]{}, pb[4]{};
        draw();
        const bool gotPa = renderer.splatDebugChecksum(pa);
        draw();
        const bool gotPb = renderer.splatDebugChecksum(pb);
        report(gotPa && gotPb && pa[0] == pb[0] && pa[1] == pb[1] && pa[2] == pb[2],
               "point mode is bit-identical across two frames");
        report(gotPa && gotA && pa[2] != a[2], "point mode composites a different frame");

        // A half mix: between the two pixel counts, and still deterministic.
        cloud->setPointMix(0.5f);
        for (int i = 0; i < 40; ++i) draw();
        const auto half = renderer.readRGBPixels();
        const long long rightHalf = countLit(half, 3 * kW / 4, kW);
        std::printf("       half mix lit px right %lld\n", rightHalf);
        report(rightHalf >= rightDots && rightHalf <= rightAfter,
               "a half mix covers between the dots and the Gaussians");

        cloud->setPointMix(0.f);
        for (int i = 0; i < 40; ++i) draw();
        const auto restored = renderer.readRGBPixels();
        const long long rightRestored = countLit(restored, 3 * kW / 4, kW);
        report(rightRestored * 50 > rightAfter * 49 && rightRestored * 49 < rightAfter * 50,
               "mix 0 brings the Gaussian frame back");
    }

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

    // The AOV has to describe THIS frame, and this frame has no splats in it.
    // The trap is that the splat pass skips itself entirely when there is
    // nothing to draw, so a clear living inside it would leave whatever the
    // last cloud-bearing frame wrote in this frame-in-flight slot — an AOV
    // reporting a cloud that is no longer in the scene, with no error anywhere.
    {
        std::vector<uint8_t> aov;
        int aw = 0, ah = 0, abpp = 0;
        if (renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                    aov, aw, ah, abpp) &&
            abpp == 4) {
            std::vector<float> d(static_cast<size_t>(aw) * ah);
            std::memcpy(d.data(), aov.data(), d.size() * sizeof(float));
            const long long covered =
                    std::count_if(d.begin(), d.end(), [](float v) { return v > 0.f; });
            std::printf("       AOV covered after deletion: %lld px\n", covered);
            report(covered == 0, "a frame with no splats leaves an EMPTY depth AOV");
        } else {
            report(false, "the depth AOV still reads back with no cloud in the scene");
        }
    }
    renderer.setSplatDepthAov(false);

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

    // The BAKE DETERMINISM gate. `second` carries the same SplatData (a
    // fixed-seed generator) uploaded into a fresh residency after the first
    // cloud's was evicted, so the two volumes are two independent runs of a
    // massively parallel scatter over identical input. They must be BYTE-
    // IDENTICAL, and they are only because the scatter accumulates with integer
    // atomicAdd: integer adds are associative, so a voxel's counters do not
    // depend on how the GPU scheduled them. A float accumulator passes every
    // visual check and fails exactly here — the same argument, and the same
    // test, particle_density.glsl's r32ui volume is built on.
    //
    // The volume is cloud-LOCAL, so `second`'s +2 in x must not move a texel
    // either; if the bake had crept into world space this is where it shows.
    std::uint64_t vol2[3]{};
    renderer.splatVolumeHash(vol2);
    std::printf("       volume  hash %llu  texels %llu  occupied %llu  gen %llu\n",
                static_cast<unsigned long long>(vol2[0]), static_cast<unsigned long long>(vol2[1]),
                static_cast<unsigned long long>(vol2[2]),
                static_cast<unsigned long long>(renderer.splatVolumeGeneration()));
    report(vol2[1] == vol1[1] && vol2[2] == vol1[2],
           "the re-uploaded cloud bakes a volume of the same extent and occupancy");
    report(vol2[0] == vol1[0],
           "with BYTE-IDENTICAL texels - the integer-atomic bake is deterministic");
    report(renderer.splatVolumeGeneration() > volGen1,
           "and evicting then re-baking bumped the volume generation");

    // A Vulkan-only cloud must never pay the GL copy: the data textures and
    // sorted-index attribute are ~1 GB at scan scale, they are CPU-side, and
    // every undo-history copy of a cloud retains them too. Lazy means a cloud
    // that has drawn plenty of Vulkan frames still has none of it.
    report(!second->glResourcesBuilt(),
           "a Vulkan-only cloud never builds its GL-side textures");

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
    // The volume is just another member of the thing retireStale already
    // manages, so eviction has to take it with the geometry and SH buffers —
    // 16 MB a cloud that would otherwise sit on the device forever.
    report(renderer.splatVolumeBytes() == 0,
           "and the reflection volume is freed with the cloud");

    // ── 5. the bake's OTHER branch: splat smaller than a voxel ──────────────
    // Everything above bakes through the footprint loop — the test cloud's
    // splats are several voxels across at 128^3 over a 3-unit box. The branch
    // that actually runs at SCAN scale is the opposite one: voxels of 10-30 cm,
    // splats of 1-5 cm, deposited as an 8-tap trilinear splat of the mean. A
    // wide, finely-grained cloud puts every splat on that path, and without it
    // the common case would ship untested.
    {
        SplatGenerator::Options o;
        o.count = 20000;
        o.seed = 909u;
        o.shDegree = 0;
        o.extent.set(8.f, 8.f, 8.f);
        o.minScale = 0.003f;
        o.maxScale = 0.006f;
        o.anisotropy = 1.f;
        o.minOpacity = 0.5f;
        o.maxOpacity = 0.95f;
        auto fine = SplatCloud::create(SplatGenerator::generate(o));
        scene->add(fine);
        for (int i = 0; i < 8; ++i) draw();

        std::uint64_t vol3[3]{};
        renderer.splatVolumeHash(vol3);
        std::printf("       fine-grained volume  texels %llu  occupied %llu\n",
                    static_cast<unsigned long long>(vol3[1]),
                    static_cast<unsigned long long>(vol3[2]));
        // 20k splats, each landing on up to 8 corners of one voxel, spread over
        // a box 8 units wide: a few thousand occupied voxels. Zero would mean
        // the sub-voxel branch deposits nothing at all.
        report(vol3[2] > 2000, "sub-voxel splats deposit through the trilinear path");

        scene->remove(*fine);
        fine.reset();
        for (int i = 0; i < 10; ++i) draw();
        report(renderer.splatVolumeBytes() == 0, "and that volume is freed too");
    }

    std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
