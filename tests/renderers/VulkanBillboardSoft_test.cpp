// VulkanBillboardSoft_test — BillboardRepr::softDistance, the soft-particle
// depth fade on ParticleField billboards.
//
// A billboard is a flat quad, depth-tested against the scene, so a big one
// whose centre sits on the ground is cut along a straight horizontal line: full
// coverage just above it, nothing just below. That line is what the knob
// removes. The scene is the smallest one that shows it: a dark ground plane, a
// camera a metre up looking slightly down, and one bright 1.2 m sprite CENTRED
// on the ground (sprite A), plus a small one high in the sky with nothing
// behind it but the sky (sprite B).
//
// Asserted, per mode (alpha-over, then additive) and per path:
//
//   1. OFF: the hard line exists. Down the column through A's centre, the
//      sprite's contribution (luma of the frame minus the same frame without
//      sprites) is near full two pixels above the line and gone below it.
//   2. ON: at the same rows the contribution is mostly gone, it grows going
//      up from the line, and a few decimetres up (where the ground behind is
//      metres away) it is the OFF value again.
//   3. B, with only sky behind it, is byte-identical on and off: the fade is
//      exactly 1.0 there and a multiply by 1.0 is exact.
//   4. Back OFF, the sprite is the hard-edged OFF sprite again (the gate).
//      That the OFF frame is the frame the renderer drew before the knob
//      existed is rg_ab's job (a build without the change, byte for byte).
//   5. The same fade in a SECONDARY view (its own G-buffer depth), in an
//      ORTHOGRAPHIC one (the other linearisation), at render scale 0.75 (the
//      fade reads the display-extent prepass, not the scaled scene), and with
//      the glow leg on (half-extent target, every second texel).
//   6. No validation errors, no render-graph diagnostics.
//
// Run twice by CTest: with the Canvas default (antialiasing 4: the overlay is
// multisampled and the depth source is a sampler2DMS) and with --aa 1 (one
// sample, sampler2D).
//
// Plain exit-code program, like the other renderer tests here. Exits 42 (CTest
// "Skipped") when no Vulkan/RT GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

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

    constexpr int kW = 640, kH = 400;
    constexpr int kSkipCode = 42;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    void setEnv(const char* k, const char* v) {
#ifdef _WIN32
        _putenv_s(k, v);
#else
        setenv(k, v, /*overwrite*/ 1);
#endif
    }

    using Pixels = std::vector<unsigned char>;

    float luma(const Pixels& img, int x, int y) {
        const std::size_t i = (std::size_t(y) * kW + std::size_t(x)) * 3u;
        return 0.2126f * img[i] + 0.7152f * img[i + 1] + 0.0722f * img[i + 2];
    }

    // The sprite's contribution down one column: |with - without| luma per
    // row, averaged over a few columns either side so ground noise cannot
    // fake a step.
    std::vector<float> profile(const Pixels& with, const Pixels& without, int x0) {
        std::vector<float> out(kH, 0.f);
        constexpr int kHalf = 3;
        for (int y = 0; y < kH; ++y) {
            float s = 0.f;
            for (int x = x0 - kHalf; x <= x0 + kHalf; ++x) s += std::abs(luma(with, x, y) - luma(without, x, y));
            out[y] = s / float(2 * kHalf + 1);
        }
        return out;
    }

    // Bytes that differ inside a pixel rectangle (inclusive).
    long long diffBytes(const Pixels& a, const Pixels& b, int x0, int y0, int x1, int y1) {
        if (a.size() != b.size() || a.empty()) return -1;
        long long d = 0;
        for (int y = std::max(y0, 0); y <= std::min(y1, kH - 1); ++y)
            for (int x = std::max(x0, 0); x <= std::min(x1, kW - 1); ++x)
                for (int c = 0; c < 3; ++c) {
                    const std::size_t i = (std::size_t(y) * kW + std::size_t(x)) * 3u + c;
                    if (a[i] != b[i]) ++d;
                }
        return d;
    }

    long long diffBytes(const Pixels& a, const Pixels& b) { return diffBytes(a, b, 0, 0, kW - 1, kH - 1); }

    float mean(const std::vector<float>& v, int lo, int hi) {
        float s = 0.f;
        int   n = 0;
        for (int i = std::max(lo, 0); i <= std::min(hi, int(v.size()) - 1); ++i, ++n) s += v[i];
        return n ? s / float(n) : 0.f;
    }

    struct LineCheck {
        bool  found = false;
        int   line  = -1;    // first row below which the OFF sprite is gone
        float ref   = 0.f;   // OFF contribution well above the line
        float offAt = 0.f;   // OFF contribution 2 px above the line
        float onAt  = 0.f;   // ON contribution at the same rows
        float onRef = 0.f;   // ON contribution well above the line
        bool  monotone = false;
    };

    // Locate the hard line in the OFF profile, searching down from a few rows
    // above where A's centre projects, and read both profiles around it.
    LineCheck measureLine(const std::vector<float>& off, const std::vector<float>& on, int projRow) {
        LineCheck c;
        c.ref = mean(off, projRow - 40, projRow - 28);
        if (c.ref < 20.f) return c;
        for (int y = projRow - 20; y < std::min(projRow + 20, kH); ++y) {
            if (off[y] < 0.25f * c.ref) {
                c.line  = y;
                c.found = true;
                break;
            }
        }
        if (!c.found) return c;
        c.offAt = mean(off, c.line - 3, c.line - 2);
        c.onAt  = mean(on, c.line - 3, c.line - 2);
        c.onRef = mean(on, projRow - 40, projRow - 28);
        // Growing away from the line. Read as the ON/OFF ratio, which is the
        // fade itself with the sprite's own radial falloff divided out (an
        // additive sprite dims going up from its centre on its own), in 3-row
        // steps, each no smaller than the one below less a little ground noise.
        c.monotone = true;
        float prev = 0.f;
        for (int y = c.line - 1; y >= c.line - 24; y -= 3) {
            const float o = mean(off, y - 2, y);
            if (o < 5.f) continue;
            const float r = mean(on, y - 2, y) / o;
            if (r + 0.06f < prev) c.monotone = false;
            prev = std::max(prev, r);
        }
        return c;
    }

    // `additive`: the bound on what is left 2 px above the line. There the fade
    // is ~6 %, and an alpha-over sprite blends in the display domain, so its
    // contribution drops to ~6 % of the OFF one. An additive sprite's radiance
    // is faded BEFORE the display transform, whose sRGB encode is steep near
    // black: 6 % of the radiance still encodes to 35 to 45 % of the levels
    // (measured; one sample lands the line a row higher than four).
    void reportLine(const char* label, const LineCheck& c, int projRow, bool additive = false) {
        std::printf("[info] %s: projected centre row %d, hard line at row %d; contribution well above "
                    "%.1f off / %.1f on, 2 px above the line %.1f off / %.1f on\n",
                    label, projRow, c.line, c.ref, c.onRef, c.offAt, c.onAt);
        check(c.found && c.ref > 20.f, std::string(label) + ": the sprite is visible and the OFF frame cuts it along a line");
        check(c.found && c.offAt > 0.6f * c.ref,
              std::string(label) + ": OFF, the sprite is near full just above the line (the hard edge)");
        check(c.found && c.onAt < (additive ? 0.6f : 0.25f) * c.offAt,
              std::string(label) + ": ON, the sprite has faded out toward the intersection");
        check(c.found && c.monotone, std::string(label) + ": ON, the coverage grows going up from the intersection");
        check(c.found && std::abs(c.onRef - c.ref) < 0.08f * c.ref + 1.5f,
              std::string(label) + ": ON, well above the ground the sprite is unchanged");
    }

}// namespace

int main(int argc, char** argv) {
    int aa = 4;// the Canvas default
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--aa") == 0 && i + 1 < argc) aa = std::atoi(argv[++i]);
    }
    std::printf("VulkanBillboardSoft_test (Canvas antialiasing %d)\n", aa);

    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(Canvas::Parameters()
                                                     .title("VulkanBillboardSoft_test")
                                                     .size(kW, kH)
                                                     .antialiasing(aa)
                                                     .vsync(false)
                                                     .headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas           = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    const auto fb = renderer.framebufferSize();
    if (fb.width() != kW || fb.height() != kH) {
        std::printf("[skip] the surface came back %dx%d, the test is laid out for %dx%d\n", fb.width(),
                    fb.height(), kW, kH);
        return kSkipCode;
    }

    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setRenderScale(1.0f);
    renderer.setAutoExposure(false);
    renderer.setClearColor(Color(0.05f, 0.06f, 0.08f));

    Scene scene;
    auto light = DirectionalLight::create(0xffffff, 2.0f);
    light->position.set(3, 8, 5);
    scene.add(light);
    auto groundMat   = MeshStandardMaterial::create();
    groundMat->color = Color(0.06f, 0.06f, 0.06f);
    groundMat->roughness = 1.f;
    auto ground = Mesh::create(PlaneGeometry::create(400.f, 400.f), groundMat);
    ground->rotateX(-math::PI / 2.f);
    scene.add(ground);

    auto camera = PerspectiveCamera::create(50.f, float(kW) / float(kH), 0.1f, 300.f);
    camera->position.set(0.f, 1.0f, 6.f);
    camera->lookAt(Vector3(0.f, 0.5f, 0.f));
    camera->updateMatrixWorld();

    // Two particles, positions in field space (= world: the field sits at the
    // origin). w is the radius.
    ParticleField::Config cfg;
    cfg.capacity      = 2;
    cfg.ownership     = ParticleField::Ownership::HostRing;
    cfg.wSemantic     = ParticleField::WSemantic::Radius;
    cfg.uniformRadius = 1.f;
    auto field = ParticleField::create(cfg);
    field->frustumCulled = false;
    const Vector3 posA(0.f, 0.f, 0.f), posB(2.4f, 2.2f, -1.f);
    const float   radA = 1.2f, radB = 0.45f;
    const float   pos[8] = {posA.x, posA.y, posA.z, radA, posB.x, posB.y, posB.z, radB};
    field->submit(pos, 2);

    const Color spriteCol(1.0f, 0.9f, 0.75f);
    field->setBillboardRepr(spriteCol, spriteCol, 1.0f);
    auto& bb        = field->billboardRepr();
    bb.brightJitter = 0.f;
    bb.softness     = 1.f;
    bb.coreWeight   = 0.f;
    bb.alphaOver    = true;
    bb.opacity      = 0.8f;
    check(bb.softDistance == 0.f, "BillboardRepr::softDistance defaults to 0 (off)");
    scene.add(field);

    const auto frame  = [&] { canvas.animateOnce([&] { renderer.render(scene, *camera); }); };
    const auto frames = [&](int n) { for (int i = 0; i < n; ++i) frame(); };

    // Where A's centre and B's quad project, in top-left pixel coordinates.
    const auto toPixel = [&](const Vector3& p, const Camera& cam) {
        Vector3 n = p;
        n.project(cam);
        return std::pair<float, float>{(n.x * 0.5f + 0.5f) * kW, (0.5f - n.y * 0.5f) * kH};
    };
    const auto [axf, ayf] = toPixel(posA, *camera);
    const int ax = int(std::lround(axf)), ay = int(std::lround(ayf));
    const auto [bxf, byf] = toPixel(posB, *camera);
    const auto [bx1f, by1f] = toPixel(posB + Vector3(radB, radB, 0.f), *camera);
    const int bHalf = int(std::ceil(std::max(std::abs(bx1f - bxf), std::abs(by1f - byf)))) + 8;
    const int bx = int(std::lround(bxf)), by = int(std::lround(byf));

    const std::uint32_t errorsBefore = vulkan::validationErrorCount();
    constexpr float kSoft = 1.5f;

    // ── The baseline: the scene without sprites ─────────────────────────────
    bb.enabled = false;
    frames(16);
    const Pixels none = renderer.readRGBPixels();
    check(none.size() == std::size_t(kW) * kH * 3u, "readRGBPixels returns the full frame");
    bb.enabled = true;

    for (const bool alphaOver : {true, false}) {
        const char* mode = alphaOver ? "alpha-over" : "additive";
        bb.alphaOver = alphaOver;
        bb.intensity = alphaOver ? 1.0f : 2.5f;

        bb.softDistance = 0.f;
        frames(12);
        const Pixels off  = renderer.readRGBPixels();
        frames(4);
        const Pixels off2 = renderer.readRGBPixels();// the frame's own run-to-run floor

        const char* capDir = std::getenv("THREEPP_SOFT_CAPTURE_DIR");
        if (capDir) renderer.writeFramebuffer(std::string(capDir) + "/" + mode + "_off.png");

        bb.softDistance = kSoft;
        frames(12);
        const Pixels on = renderer.readRGBPixels();
        if (capDir) renderer.writeFramebuffer(std::string(capDir) + "/" + mode + "_on.png");

        bb.softDistance = 0.f;
        frames(12);
        const Pixels offAgain = renderer.readRGBPixels();

        const auto pOff = profile(off, none, ax);
        const LineCheck c = measureLine(pOff, profile(on, none, ax), ay);
        reportLine(mode, c, ay, !alphaOver);

        const long long floorB = diffBytes(off, off2, bx - bHalf, by - bHalf, bx + bHalf, by + bHalf);
        const long long bDiff  = diffBytes(off, on, bx - bHalf, by - bHalf, bx + bHalf, by + bHalf);
        std::printf("[info] %s: sprite B (sky behind) on vs off %lld bytes (floor %lld)\n", mode, bDiff,
                    floorB);
        check(bDiff >= 0 && bDiff <= floorB,
              std::string(mode) + ": a sprite with only sky behind it is byte-identical with the fade on");

        // The gate, on the SPRITE: its contribution down A's column is the OFF
        // one again. Not the whole frame's bytes — the ground under it is lit
        // by temporal GI that keeps moving by a level here and there from one
        // frame index to the next (two OFF frames 4 apart already differ),
        // while a sprite composited after the temporal resolve has no history
        // the fade could have left behind.
        const auto pAgain = profile(offAgain, none, ax);
        float worst = 0.f;
        for (int y = std::max(ay - 80, 0); y < std::min(ay + 20, kH); ++y)
            worst = std::max(worst, std::abs(pAgain[y] - pOff[y]));
        std::printf("[info] %s: back at 0, largest change of the sprite's contribution down its column "
                    "%.2f levels (frame bytes %lld, vs %lld between two OFF frames 4 apart)\n",
                    mode, worst, diffBytes(off, offAgain), diffBytes(off, off2));
        check(worst < 2.f, std::string(mode) + ": softDistance back at 0 gives the hard-edged sprite again");
    }

    // ── The glow leg ─────────────────────────────────────────────────────────
    // Additive with a halo: the half-extent glow target reads every second
    // texel of the same depth buffer. The halo is blurred, so this asserts the
    // halo FADES near the line too (the summed glow below A's centre drops)
    // rather than a profile.
    {
        bb.alphaOver = false;
        bb.intensity = 2.5f;
        bb.glow      = 4.f;
        bb.softDistance = 0.f;
        frames(12);
        const Pixels glowOff = renderer.readRGBPixels();
        bb.softDistance = kSoft;
        frames(12);
        const Pixels glowOn = renderer.readRGBPixels();
        // Below the line the sharp sprite is depth-tested away in both frames,
        // so what is left there is the halo spilling over the edge.
        float spillOff = 0.f, spillOn = 0.f;
        for (int y = ay + 4; y < ay + 24; ++y)
            for (int x = ax - 40; x <= ax + 40; ++x) {
                spillOff += std::abs(luma(glowOff, x, y) - luma(none, x, y));
                spillOn += std::abs(luma(glowOn, x, y) - luma(none, x, y));
            }
        std::printf("[info] glow: halo spill below the line %.0f off / %.0f on\n", spillOff, spillOn);
        check(spillOff > 0.f && spillOn < spillOff,
              "glow leg: the halo below the intersection fades with the sprite");
        bb.glow         = 0.f;
        bb.softDistance = 0.f;
        bb.alphaOver    = true;
        bb.intensity    = 1.0f;
    }

    // ── Render scale 0.75 ────────────────────────────────────────────────────
    // The scene renders at 480x300 and is upsampled; the overlay depth the
    // fade reads is the display-extent prepass, so the ramp is where it was.
    {
        renderer.setRenderScale(0.75f);
        bb.enabled = false;
        frames(16);
        const Pixels noneS = renderer.readRGBPixels();
        bb.enabled      = true;
        bb.softDistance = 0.f;
        frames(12);
        const Pixels offS = renderer.readRGBPixels();
        bb.softDistance = kSoft;
        frames(12);
        const Pixels onS = renderer.readRGBPixels();
        reportLine("render scale 0.75", measureLine(profile(offS, noneS, ax), profile(onS, noneS, ax), ay), ay);
        bb.softDistance = 0.f;
        renderer.setRenderScale(1.0f);
        frames(4);
    }

    // ── Secondary views: perspective, then orthographic ─────────────────────
    // Each view's sprites are tested against its OWN G-buffer depth, so each
    // gets its own linear-depth pass; the orthographic one exercises the other
    // linearisation (w row constant).
    {
        auto secCam = PerspectiveCamera::create(50.f, float(kW) / float(kH), 0.1f, 300.f);
        secCam->position.copy(camera->position);
        secCam->lookAt(Vector3(0.f, 0.5f, 0.f));
        secCam->updateMatrixWorld();
        auto orthoCam = OrthographicCamera::create(-2.4f, 2.4f, 1.5f, -1.5f, 0.1f, 300.f);
        orthoCam->position.set(0.f, 1.0f, 6.f);
        orthoCam->lookAt(Vector3(0.f, 0.5f, 0.f));
        orthoCam->updateMatrixWorld();

        struct V {
            const char* label;
            Camera*     cam;
        };
        for (const V& v : {V{"secondary view", secCam.get()}, V{"orthographic view", orthoCam.get()}}) {
            const std::uint32_t h = renderer.addView(*v.cam, kW, kH);
            if (h == 0u) {
                check(false, std::string(v.label) + ": addView returned a handle");
                continue;
            }
            const auto [vxf, vyf] = toPixel(posA, *v.cam);
            const int vx = int(std::lround(vxf)), vy = int(std::lround(vyf));
            bb.enabled = false;
            frames(16);
            const Pixels vNone = renderer.readViewRGBPixels(h);
            bb.enabled      = true;
            bb.softDistance = 0.f;
            frames(12);
            const Pixels vOff = renderer.readViewRGBPixels(h);
            bb.softDistance = kSoft;
            frames(12);
            const Pixels vOn = renderer.readViewRGBPixels(h);
            bb.softDistance = 0.f;
            if (vNone.size() != none.size() || vOff.size() != none.size() || vOn.size() != none.size()) {
                check(false, std::string(v.label) + ": readViewRGBPixels returns the full view");
            } else {
                reportLine(v.label, measureLine(profile(vOff, vNone, vx), profile(vOn, vNone, vx), vy), vy);
            }
            (void) renderer.removeView(h);
            frames(4);
        }
    }

    // ── Validation and the frame graph ───────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       assertion is skipped (it would be vacuously true).\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "soft billboards provoke no validation errors");
    }
    for (const auto& m : renderer.renderGraphDiagnostics()) std::printf("  [graph] %s\n", m.c_str());
    check(renderer.renderGraphDiagnosticCount() == 0, "soft billboards produce no render-graph diagnostics");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
