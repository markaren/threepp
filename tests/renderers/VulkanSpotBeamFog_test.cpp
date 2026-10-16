// VulkanSpotBeamFog_test — a strong spot light's beam in fog renders smooth,
// settles under the temporal resolve, and keeps its brightness through it.
//
// The scene is the one that found the bug (the dune buggy's drive beam seen
// from the cockpit): exp2 fog, a 3000-unit spot light with a 0.09 rad half
// angle and a 200 m range aimed at the ground 150 m ahead, and a camera 0.4 m
// below and 0.35 m beside the lamp looking down the beam. Near the lamp the
// beam is centimetres across, so the view rays above the horizon cross it in a
// short stretch close to the lamp, where the in-scatter peaks as the inverse
// square of the distance. volumetricSpotScatter used to spread its steps evenly
// over the light's whole range sphere and hit or missed that stretch at random:
// a sparse spray of very bright samples per frame, which the temporal resolve
// neither averages (its clip throws the bright ones away, so the beam comes
// out dark) nor hides (what survives is grain that never settles). It now clips
// the march to the cone and samples it equi-angularly about the lamp.
//
// Tone mapping is off, so a displayed pixel decodes back to linear light.
//
//   1. Per frame: the linear scene image the shade wrote (before any temporal
//      resolve), over 24 frames: in the beam, a pixel's frame-to-frame standard
//      deviation is small next to its mean. (The old march: median 0.49 of
//      the mean, p90 1.3; now 0.004 and 0.013.)
//   2. Settled under TAA (48 still frames): the beam's pixel-scale noise is no
//      larger than the clear sky's beside it. (The old march: p99 11.6, max 17
//      levels; now 0.6 and 1.)
//   3. The resolve keeps the beam: settled under TAA, the beam is as bright as
//      the mean of the frames the shade wrote, within 5 % (the old march: 0.79
//      of it, the clip threw its bright samples away; now 1.02), and as bright
//      as settled under FSR where FSR is available.
//   4. No validation errors, no render-graph diagnostics.
//
// Plain exit-code program, like the other renderer tests here. Exits 42 (CTest
// "Skipped") when no Vulkan/RT GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/FogExp2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    // The air: rows above the horizon (which sits near row 190), clear of the
    // top edge. The sky reference: the left columns, which the beam (it rises
    // to the right) never reaches.
    constexpr int kAirY0 = 8, kAirY1 = 165;
    constexpr int kSkyX0 = 8, kSkyX1 = 120;

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

    using LumaImage = std::vector<float>;// one value per pixel, top-left origin

    float halfToFloat(std::uint16_t h) {
        const std::uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
        float v;
        if (exp == 0) v = std::ldexp(float(man), -24);
        else if (exp == 31) v = man ? NAN : INFINITY;
        else v = std::ldexp(float(man | 0x400u), int(exp) - 25);
        return sign ? -v : v;
    }

    float srgbToLinear(float c) {
        c /= 255.f;
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }

    // Linear luminance of the displayed frame.
    LumaImage displayLinear(const std::vector<unsigned char>& rgb) {
        LumaImage out(std::size_t(kW) * kH);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = 0.2126f * srgbToLinear(rgb[i * 3]) + 0.7152f * srgbToLinear(rgb[i * 3 + 1]) +
                     0.0722f * srgbToLinear(rgb[i * 3 + 2]);
        return out;
    }

    // 8-bit display luma (what the eye reads as grain).
    LumaImage display8(const std::vector<unsigned char>& rgb) {
        LumaImage out(std::size_t(kW) * kH);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = 0.2126f * rgb[i * 3] + 0.7152f * rgb[i * 3 + 1] + 0.0722f * rgb[i * 3 + 2];
        return out;
    }

    // |pixel - its 5x5 box mean|: pixel-scale noise, blind to smooth gradients.
    LumaImage highPass(const LumaImage& v) {
        LumaImage out(v.size(), 0.f);
        for (int y = 2; y < kH - 2; ++y)
            for (int x = 2; x < kW - 2; ++x) {
                float s = 0.f;
                for (int dy = -2; dy <= 2; ++dy)
                    for (int dx = -2; dx <= 2; ++dx) s += v[std::size_t(y + dy) * kW + (x + dx)];
                out[std::size_t(y) * kW + x] = std::abs(v[std::size_t(y) * kW + x] - s / 25.f);
            }
        return out;
    }

    // Per air row, the clear sky's mean: what the beam is measured against.
    std::vector<float> skyRows(const LumaImage& v) {
        std::vector<float> out(kH, 0.f);
        for (int y = kAirY0; y <= kAirY1; ++y) {
            float s = 0.f;
            for (int x = kSkyX0; x <= kSkyX1; ++x) s += v[std::size_t(y) * kW + x];
            out[y] = s / float(kSkyX1 - kSkyX0 + 1);
        }
        return out;
    }

    float percentile(std::vector<float> v, float p) {
        if (v.empty()) return 0.f;
        const std::size_t k = std::min(v.size() - 1, std::size_t(p / 100.f * float(v.size() - 1) + 0.5f));
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
        return v[k];
    }

    // The beam's mean excess over the sky, over the given pixels.
    float beamExcess(const LumaImage& v, const std::vector<std::size_t>& beam) {
        const auto sky = skyRows(v);
        double s       = 0.0;
        for (const std::size_t i : beam) s += v[i] - sky[i / kW];
        return beam.empty() ? 0.f : float(s / double(beam.size()));
    }

}// namespace

int main() {
    std::printf("VulkanSpotBeamFog_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");
    // The per-frame linear scene image (readSceneHdrDebug) needs its own
    // allocation to be readable after the frame.
    setEnv("THREEPP_VK_NO_ALIAS", "1");

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanSpotBeamFog_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas           = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;
    const auto fb            = renderer.framebufferSize();
    if (fb.width() != kW || fb.height() != kH) {
        std::printf("[skip] the surface came back %dx%d, the test is laid out for %dx%d\n", fb.width(),
                    fb.height(), kW, kH);
        return kSkipCode;
    }

    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setRenderScale(1.0f);
    renderer.setAutoExposure(false);
    renderer.toneMapping         = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;
    renderer.setFogAnisotropy(0.45f);

    Scene scene;
    scene.background = Color(0.01f, 0.012f, 0.02f);
    scene.fog        = FogExp2(Color(0.42f, 0.42f, 0.44f), 0.0010f);
    auto ground      = Mesh::create(PlaneGeometry::create(600.f, 600.f), MeshStandardMaterial::create());
    ground->rotateX(-math::PI / 2.f);
    scene.add(ground);

    const float lampY = 1.65f;
    auto spot         = SpotLight::create(Color(1.0f, 0.86f, 0.68f), 3000.f, 200.f, 0.09f, 0.6f, 1.0f);
    spot->radius      = 0.3f;
    spot->position.set(0.f, lampY, 0.3f);
    Object3D target;
    target.position.set(0.f, lampY - lampY / 150.f * 10.f, 10.3f);// the axis meets the ground 150 m ahead
    spot->setTarget(target);
    scene.add(spot);

    auto camera = PerspectiveCamera::create(62.f, float(kW) / float(kH), 0.1f, 6000.f);
    camera->position.set(0.35f, lampY - 0.4f, -0.15f);
    camera->lookAt(Vector3(0.35f, lampY - 0.9f, 20.f));
    camera->updateMatrixWorld();

    double t         = 0.0;
    const auto frame = [&] {
        t += 1.0 / 60.0;
        renderer.setSimTime(t);
        canvas.animateOnce([&] { renderer.render(scene, *camera); });
    };
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    // ── 1. Per frame: the shade's own linear image ──────────────────────────
    for (int i = 0; i < 16; ++i) frame();// pipelines, froxel history
    constexpr int kFrames = 24;
    std::vector<LumaImage> frames;
    for (int i = 0; i < kFrames; ++i) {
        frame();
        std::vector<std::uint8_t> raw;
        int w = 0, h = 0;
        if (!renderer.readSceneHdrDebug(raw, w, h) || w != kW || h != kH) {
            std::printf("[info] no readable scene image (%dx%d): the per-frame check is skipped\n", w, h);
            frames.clear();
            break;
        }
        LumaImage lum(std::size_t(kW) * kH);
        for (std::size_t p = 0; p < lum.size(); ++p) {
            std::uint16_t c[4];
            std::memcpy(c, raw.data() + p * 8, 8);
            lum[p] = 0.2126f * halfToFloat(c[0]) + 0.7152f * halfToFloat(c[1]) + 0.0722f * halfToFloat(c[2]);
        }
        frames.push_back(std::move(lum));
    }

    // ── 2. Settled under TAA ────────────────────────────────────────────────
    for (int i = 0; i < 48; ++i) frame();
    const auto rgbTaa  = renderer.readRGBPixels();
    const LumaImage taaLin = displayLinear(rgbTaa);
    if (const char* dir = std::getenv("THREEPP_SPOTBEAM_CAPTURE_DIR")) {
        renderer.writeFramebuffer(std::string(dir) + "/spot_beam_taa.png");
    }

    // The beam: air pixels whose settled excess over the sky is at least a
    // fifth of the 99th percentile excess (robust to a stray bright pixel).
    std::vector<std::size_t> beam;
    {
        const auto sky = skyRows(taaLin);
        std::vector<float> ex;
        for (int y = kAirY0; y <= kAirY1; ++y)
            for (int x = kSkyX1 + 1; x < kW - 2; ++x) ex.push_back(taaLin[std::size_t(y) * kW + x] - sky[y]);
        const float ref = percentile(ex, 99.f);
        for (int y = kAirY0; y <= kAirY1; ++y)
            for (int x = kSkyX1 + 1; x < kW - 2; ++x)
                if (taaLin[std::size_t(y) * kW + x] - sky[y] >= 0.2f * ref) beam.push_back(std::size_t(y) * kW + x);
        std::printf("[info] beam: %zu pixels, p99 excess %.5f (linear)\n", beam.size(), ref);
        check(beam.size() > 1500 && ref > 0.002f, "the beam shows in the fog above the horizon");
    }
    if (beam.empty()) {
        std::printf("FAIL (no beam)\n");
        return 1;
    }

    if (!frames.empty()) {
        std::vector<float> rel;
        for (const std::size_t i : beam) {
            double m = 0.0, v = 0.0;
            for (const auto& f : frames) m += f[i];
            m /= kFrames;
            for (const auto& f : frames) v += (f[i] - m) * (f[i] - m);
            v /= (kFrames - 1);
            if (m > 0.0) rel.push_back(float(std::sqrt(v) / m));
        }
        const float med = percentile(rel, 50.f), p90 = percentile(rel, 90.f);
        std::printf("[info] per frame, a beam pixel's standard deviation over its mean: median %.3f, p90 %.3f\n",
                    med, p90);
        check(med < 0.1f && p90 < 0.25f,
              "per frame, the beam is smooth (a pixel's deviation under a tenth of its mean, p90 a quarter)");
    }

    const LumaImage hp = highPass(display8(rgbTaa));
    std::vector<float> beamHp, skyHp;
    for (const std::size_t i : beam) beamHp.push_back(hp[i]);
    for (int y = kAirY0; y <= kAirY1; ++y)
        for (int x = kSkyX0; x <= kSkyX1; ++x) skyHp.push_back(hp[std::size_t(y) * kW + x]);
    const float beamP99 = percentile(beamHp, 99.f), skyP99 = percentile(skyHp, 99.f);
    const float beamMax = *std::max_element(beamHp.begin(), beamHp.end());
    std::printf("[info] settled under TAA: pixel noise p99 %.2f max %.1f in the beam, p99 %.2f in the sky (8-bit)\n",
                beamP99, beamMax, skyP99);
    check(beamP99 <= skyP99 + 1.0f && beamMax <= 4.f,
          "settled, the beam's pixel noise is no larger than the clear sky's (p99 within a level, max 4)");

    // ── 3. The resolve keeps the beam ───────────────────────────────────────
    const float taaBeam = beamExcess(taaLin, beam);
    if (!frames.empty()) {
        LumaImage meanHdr(frames[0].size(), 0.f);
        for (const auto& f : frames)
            for (std::size_t i = 0; i < f.size(); ++i) meanHdr[i] += f[i] / float(kFrames);
        const float hdrBeam = beamExcess(meanHdr, beam);
        std::printf("[info] beam excess: the frames' mean %.5f, settled under TAA %.5f (ratio %.3f)\n", hdrBeam,
                    taaBeam, taaBeam / hdrBeam);
        check(std::abs(taaBeam / hdrBeam - 1.f) < 0.05f,
              "settled under TAA, the beam is as bright as the frames the shade wrote (within 5 %)");
    }
    if (renderer.fsrAvailable()) {
        renderer.setFsr(true);
        renderer.setRenderScale(1.0f);
        for (int i = 0; i < 48; ++i) frame();
        const float fsrBeam = beamExcess(displayLinear(renderer.readRGBPixels()), beam);
        std::printf("[info] settled beam excess: TAA %.5f, FSR %.5f (ratio %.3f)\n", taaBeam, fsrBeam,
                    taaBeam / fsrBeam);
        check(std::abs(taaBeam / fsrBeam - 1.f) < 0.05f,
              "settled, the beam is as bright through TAA as through FSR (within 5 %)");
        renderer.setFsr(false);
    } else {
        std::printf("[note] FSR is not available: the TAA/FSR brightness check is skipped\n");
    }

    // ── 4. Validation and the frame graph ───────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       assertion is skipped (it would be vacuously true).\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the spot beam in fog provokes no validation errors");
    }
    for (const auto& m : renderer.renderGraphDiagnostics()) std::printf("  [graph] %s\n", m.c_str());
    check(renderer.renderGraphDiagnosticCount() == 0, "the spot beam in fog produces no render-graph diagnostics");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
