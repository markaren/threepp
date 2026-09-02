// SplatOverlayOcclusion_probe — does a splat cloud hide the overlay FILL path?
//
// The stamp (splat_overlay_depth.frag) exists so that post-resolve overlay
// draws — lines, wireframe, world sprites, AND the kSnapUiBlend fill route an
// untextured transparent MeshBasicMaterial takes — are depth-tested against
// the splat cloud that stands in front of them. The AOV the stamp reads is
// latched ON automatically the first frame a scene holds both clouds and
// overlay content (VulkanCoreScene.cpp, "Overlay occlusion latch").
//
// This probe measures that latch on the fill path. A wall of splats covers the
// left half of the frame; a bright transparent box sits BEHIND it, spanning
// from behind the wall out into open space. The half behind the wall must
// disappear; the half beside it must not.
//
//   SplatOverlayOcclusion_probe            the latch does it (the thing under test)
//   SplatOverlayOcclusion_probe --explicit setSplatDepthAov(Median) first: the
//                                          control that isolates the latch from
//                                          the stamp
//   SplatOverlayOcclusion_probe --line     a LineSegments overlay instead of the
//                                          fill mesh (the path known to work)
//   SplatOverlayOcclusion_probe --lod      submit ranges set per frame, the way
//                                          splats::selectLod drives a cloud
//                                          loaded by loadSogWithLod
//   SplatOverlayOcclusion_probe --pip      a second (secondary) view attached,
//                                          the demo's picture-in-picture
//   SplatOverlayOcclusion_probe --scale X the demo renders at render_scale 0.5
//   SplatOverlayOcclusion_probe --dump     write the two frames as PPMs
//
// Exits 42 without a Vulkan GPU, like the other Vulkan probes. Exit 1 if the
// behind-the-cloud half of the overlay survives.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    constexpr int kWarmFrames = 24;

    // A dense, opaque grey wall at z = 0 covering x in [-1.1, 1.1]. Opaque
    // enough that the AOV's coverage > 0.5 gate is satisfied everywhere the
    // wall is drawn — the stamp is binary at that gate, so a wispy cloud would
    // measure the gate rather than the latch.
    SplatData makeWall() {
        SplatGenerator::Options o;
        o.count = 30000;
        o.seed = 11u;
        o.shDegree = 0;
        o.extent.set(1.1f, 1.6f, 0.02f);
        o.minScale = 0.03f;
        o.maxScale = 0.05f;
        o.anisotropy = 1.1f;
        o.minOpacity = 0.9f;
        o.maxOpacity = 1.f;
        auto d = SplatGenerator::generate(o);
        for (size_t i = 0; i < d.count(); ++i) d.setDcColor(i, Vector3{0.55f, 0.55f, 0.58f});
        return d;
    }

    void writePPM(const fs::path& p, const std::vector<unsigned char>& rgb) {
        std::ofstream f(p, std::ios::binary);
        f << "P6\n" << kW << " " << kH << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    }

    // The overlay is the only saturated-magenta thing in the frame: the wall is
    // neutral grey and the background is dark blue-grey.
    bool isOverlayPixel(const std::vector<unsigned char>& rgb, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * kW + x) * 3;
        const int r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
        return r > 90 && b > 90 && g + 40 < r && g + 40 < b;
    }

    long long countOverlay(const std::vector<unsigned char>& rgb, int x0, int x1) {
        long long n = 0;
        for (int y = 0; y < kH; ++y)
            for (int x = x0; x < x1; ++x)
                if (isOverlayPixel(rgb, x, y)) ++n;
        return n;
    }

}// namespace

int main(int argc, char** argv) {
    bool explicitAov = false, useLine = false, dump = false, useLod = false, usePip = false;
    bool lateOverlay = false;
    double renderScale = 1.0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--explicit") == 0) explicitAov = true;
        else if (std::strcmp(argv[i], "--line") == 0) useLine = true;
        else if (std::strcmp(argv[i], "--lod") == 0) useLod = true;
        else if (std::strcmp(argv[i], "--pip") == 0) usePip = true;
        else if (std::strcmp(argv[i], "--late") == 0) lateOverlay = true;
        else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) renderScale = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--dump") == 0) dump = true;
    }

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(Canvas::Parameters()
                                                     .title("SplatOverlayOcclusion_probe")
                                                     .size(kW, kH)
                                                     .vsync(false)
                                                     .headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;

    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.f;
    renderer.setRenderScale(static_cast<float>(renderScale));
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setGbufferMsaa(2);// unjittered raster: the stamp compares against it
    renderer.setClearColor(Color(0.05f, 0.05f, 0.09f));
    if (explicitAov) renderer.setSplatDepthAov(VulkanRenderer::SplatDepthMode::Median);

    auto scene = Scene::create();
    scene->background = Color(0x0d0d17);

    auto camera = PerspectiveCamera::create(50, static_cast<float>(kW) / kH, 0.05f, 100.f);
    camera->position.set(0.f, 0.f, 5.f);
    camera->lookAt(Vector3{0.f, 0.f, 0.f});

    // The overlay: an untextured TRANSPARENT MeshBasicMaterial, which is
    // exactly what snapMeshFlags routes down the kSnapUiBlend fill path. It
    // sits at z = -1.2, BEHIND the wall, and spans x in [-0.4, 2.6]: the left
    // part is behind the wall, the right part is in open space beside it.
    auto overlayMat = MeshBasicMaterial::create();
    overlayMat->color = Color(0xff20ff);
    overlayMat->transparent = true;
    overlayMat->opacity = 0.95f;
    auto box = Mesh::create(BoxGeometry::create(3.f, 0.8f, 0.05f), overlayMat);
    box->position.set(1.1f, 0.f, -1.2f);

    std::shared_ptr<LineSegments> lines;
    if (useLine) {
        auto lg = BufferGeometry::create();
        std::vector<float> v;
        for (int i = 0; i <= 24; ++i) {
            const float x = -0.4f + 3.f * static_cast<float>(i) / 24.f;
            v.insert(v.end(), {x, -0.4f, -1.2f, x, 0.4f, -1.2f});
        }
        lg->setAttribute("position", FloatBufferAttribute::create(v, 3));
        auto lm = LineBasicMaterial::create();
        lm->color = Color(0xff20ff);
        lines = LineSegments::create(lg, lm);
    }

    auto cloud = SplatCloud::create(makeWall());

    // The PiP the demo attaches: a persistent secondary view. The stamp is
    // primary-only by scope, so this must not change the primary's answer.
    auto pipCamera = PerspectiveCamera::create(50, 1.f, 0.05f, 100.f);
    pipCamera->position.set(3.f, 1.f, 4.f);
    pipCamera->lookAt(Vector3{0.f, 0.f, 0.f});
    if (usePip) renderer.addView(*pipCamera, 128, 128);

    auto draw = [&] {
        // What splats::selectLod does every frame on a loadSogWithLod cloud:
        // rewrite the submit ranges. Here they cover the whole cloud in four
        // ascending chunks, which is the identity by construction.
        if (useLod) {
            const auto total = static_cast<uint32_t>(cloud->splatCount());
            const uint32_t per = (total + 3u) / 4u;
            std::vector<std::pair<uint32_t, uint32_t>> r;
            for (uint32_t off = 0; off < total; off += per)
                r.emplace_back(off, std::min(per, total - off));
            cloud->setSubmitRanges(std::move(r));
        }
        renderer.render(*scene, *camera);
    };
    auto capture = [&](int frames) {
        for (int i = 0; i < frames; ++i) draw();
        return renderer.readRGBPixels();
    };

    // ── A: the overlay alone, no cloud ─────────────────────────────────────
    if (useLine) scene->add(lines);
    else scene->add(box);
    const auto noCloud = capture(kWarmFrames);

    // ── B: the same overlay with the wall in front of its left half ────────
    // --late is the demo's order: the cloud is resident for many frames BEFORE
    // any overlay content joins the scene, so the latch has to fire on a later
    // frame rather than on the cloud's first.
    if (lateOverlay) {
        if (useLine) scene->remove(*lines);
        else scene->remove(*box);
        scene->add(cloud);
        for (int i = 0; i < kWarmFrames; ++i) draw();
        if (useLine) scene->add(lines);
        else scene->add(box);
    } else {
        scene->add(cloud);
    }
    const auto withCloud = capture(kWarmFrames * 3);

    if (dump) {
        writePPM(fs::path("splat_overlay_nocloud.ppm"), noCloud);
        writePPM(fs::path("splat_overlay_withcloud.ppm"), withCloud);
    }

    // Column 192 is the screen centre. The wall's drawn silhouette ends near
    // column 230; the overlay bar runs to column ~307. Measure a band that is
    // strictly behind the wall against one that is strictly beside it, with a
    // margin on each side of the silhouette edge.
    const long long behindA = countOverlay(noCloud, 175, 225);
    const long long behindB = countOverlay(withCloud, 175, 225);
    const long long besideA = countOverlay(noCloud, 245, 300);
    const long long besideB = countOverlay(withCloud, 245, 300);

    std::printf("mode           : %s%s%s%s\n", useLine ? "LineSegments overlay" : "kSnapUiBlend fill mesh",
                explicitAov ? " (+ explicit setSplatDepthAov)" : "",
                useLod ? " (+ per-frame submit ranges)" : "",
                usePip ? " (+ secondary PiP view)" : "");
    if (lateOverlay) std::printf("                 (+ overlay added AFTER the cloud was resident)\n");
    // The diagnostic that used to lie: with the AOV allocated by the latch
    // alone, splatDepthAov() is false and the readback used to gate on it, so
    // this read came back empty on exactly the frames where the stamp WAS
    // running. It must succeed at full resolution now.
    std::vector<uint8_t> aovBytes;
    int aw = 0, ah = 0, abpp = 0;
    const bool aovRead = renderer.readGBufferAOV(VulkanRenderer::GBufferAOV::SplatDepth,
                                                 aovBytes, aw, ah, abpp);
    std::printf("splat_depth_aov: asked=%s mode=%d  readback=%s %dx%d\n",
                renderer.splatDepthAov() ? "yes" : "no",
                static_cast<int>(renderer.splatDepthAovMode()),
                aovRead ? "OK" : "EMPTY", aw, ah);
    std::printf("behind cloud   : %lld px without cloud -> %lld px with cloud\n", behindA, behindB);
    std::printf("beside cloud   : %lld px without cloud -> %lld px with cloud\n", besideA, besideB);

    if (behindA < 200 || besideA < 200) {
        std::printf("[FAIL] fixture: the overlay does not cover both bands (%lld / %lld)\n",
                    behindA, besideA);
        return 1;
    }
    const double hidden = 1.0 - static_cast<double>(behindB) / static_cast<double>(behindA);
    const double kept = static_cast<double>(besideB) / static_cast<double>(besideA);
    std::printf("occluded       : %.1f%% of the behind-band hidden, %.1f%% of the beside-band kept\n",
                100.0 * hidden, 100.0 * kept);

    if (hidden < 0.9) {
        std::printf("[FAIL] the cloud does NOT occlude the overlay behind it\n");
        return 1;
    }
    if (kept < 0.9) {
        std::printf("[FAIL] the cloud occludes the overlay BESIDE it too\n");
        return 1;
    }
    // At the RENDER extent, not the display one — every G-buffer AOV is.
    const int expectW = static_cast<int>(kW * renderScale + 0.5);
    if (!aovRead || aw != expectW) {
        std::printf("[FAIL] the stamp ran but the splat_depth AOV reads back as %s (%dx%d, "
                    "expected width %d)\n",
                    aovRead ? "the wrong size" : "unallocated", aw, ah, expectW);
        return 1;
    }
    std::printf("[ ok ] cloud occludes the overlay behind it and leaves the rest alone\n");
    return 0;
}
