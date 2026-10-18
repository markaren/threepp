// VulkanGrassMotion_test — a GrassMesh's motion vectors are its sway since the
// last frame, and zero when the field holds still.
//
// grass_wind.comp writes each frame's displaced pose straight into the field's
// BLAS vertex buffer, but nothing ever refreshed blas->prevVertex, the previous
// pose the G-buffer and the closest-hit reprojection read: it kept the pose of
// the BLAS build. So every grass pixel reported its offset from that pose, not
// its motion: a field frozen in time read up to a pixel of motion, a swaying
// one the wrong amount, and TAA and the upscalers (DLSS, FSR) reprojected the
// grass's history to the wrong place every frame (found in a pine forest scene
// whose grass smeared under DLSS). Each deform now copies the shown pose into
// prevVertex first, the prime leaves prev == vertex, and a field the distance
// gate freezes gets that copy once.
//
// The scene: nine thin blades in a row 3 m in front of a still camera, wind
// along screen x, nothing else. Motion is read from the Motion AOV
// (prevNDC - currNDC), grass pixels from the Ids AOV.
//
//   1. Time frozen (the scene clock does not move, at two different moments
//      of the gust): no grass pixel reports motion.
//   2. One step of 0.1 s: the top of the blades reports the analytic sway
//      step of grass_wind.comp, sign and size (within 10 %), and no vertical
//      motion; the empty background reports none.
//   3. The distance gate freezes the field after it swayed: its motion
//      collapses to zero the next frame.
//   4. No validation errors.
//
// Plain exit-code program, like the other renderer tests here. Exits 42 (CTest
// "Skipped") when no Vulkan/RT GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 480, kH = 320;
    constexpr float kFovDeg = 50.f;
    constexpr float kCamDist = 3.f;      // camera z; the blades stand at z = 0
    constexpr float kCamY = 0.6f;
    constexpr float kWind = 0.3f;        // windStrength
    constexpr float kBladeH = 1.2f;
    constexpr float kBladeW = 0.06f;
    constexpr int kSegments = 4;         // rows of vertices per blade: kSegments + 1
    constexpr float kZero = 1e-3f;       // NDC; a pixel here is 2/480 = 0.0042

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

    // grass_wind.comp's gust, for a blade rooted at (rx, rz).
    float gust(float t, float rx, float rz) {
        const float phase = t * 1.6f + rx * 0.25f + rz * 0.25f;
        return std::sin(phase) * 0.6f + std::sin(phase * 2.3f + 1.7f) * 0.25f;
    }

    std::vector<float> bladeXs() {
        std::vector<float> xs;
        for (int i = 0; i < 9; ++i) xs.push_back(-0.8f + 0.2f * float(i));
        return xs;
    }

    std::shared_ptr<BufferGeometry> makeBlades() {
        std::vector<float> pos, nrm, uv, hf;
        std::vector<unsigned int> idx;
        for (float x : bladeXs()) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            for (int s = 0; s <= kSegments; ++s) {
                const float f = float(s) / float(kSegments);
                for (float dx : {-0.5f * kBladeW, 0.5f * kBladeW}) {
                    pos.insert(pos.end(), {x + dx, f * kBladeH, 0.f});
                    nrm.insert(nrm.end(), {0.f, 0.f, 1.f});
                    uv.insert(uv.end(), {dx > 0.f ? 1.f : 0.f, f});
                    hf.push_back(f);
                }
            }
            for (unsigned int s = 0; s < unsigned(kSegments); ++s) {
                const unsigned int a = base + 2 * s, b = a + 1, c = a + 2, d = a + 3;
                idx.insert(idx.end(), {a, b, d, a, d, c});
            }
        }
        auto g = BufferGeometry::create();
        g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        g->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        g->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        g->setAttribute("heightFrac", FloatBufferAttribute::create(hf, 1));
        g->setIndex(idx);
        return g;
    }

    float halfToFloat(std::uint16_t v) {
        const std::uint32_t sign = std::uint32_t(v >> 15) << 31;
        const std::int32_t exp = (v >> 10) & 0x1F;
        const std::uint32_t man = v & 0x3FF;
        std::uint32_t bits;
        if (exp == 0) {
            float f = float(man) * 5.9604645e-8f;
            std::memcpy(&bits, &f, 4);
            bits |= sign;
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (man << 13);
        } else {
            bits = sign | (std::uint32_t(exp - 15 + 127) << 23) | (man << 13);
        }
        float out;
        std::memcpy(&out, &bits, 4);
        return out;
    }

    struct Frame {
        int w = 0, h = 0;
        std::vector<float> mx, my;  // motion, NDC (prev - curr)
        std::vector<bool> grass;    // Ids AOV: an instance was hit
    };

    bool readFrame(VulkanRenderer& r, Frame& f) {
        std::vector<std::uint8_t> raw;
        int bw = 0, bh = 0, bpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Motion, raw, bw, bh, bpp) || bpp != 8) return false;
        f.w = bw;
        f.h = bh;
        const std::size_t n = std::size_t(bw) * std::size_t(bh);
        f.mx.resize(n);
        f.my.resize(n);
        const auto* hf = reinterpret_cast<const std::uint16_t*>(raw.data());
        for (std::size_t i = 0; i < n; ++i) {
            f.mx[i] = halfToFloat(hf[i * 4 + 0]);
            f.my[i] = halfToFloat(hf[i * 4 + 1]);
        }
        int iw = 0, ih = 0, ibpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, iw, ih, ibpp) || iw != bw || ih != bh ||
            ibpp != 16)
            return false;
        const auto* ids = reinterpret_cast<const std::uint32_t*>(raw.data());
        f.grass.resize(n);
        for (std::size_t i = 0; i < n; ++i) f.grass[i] = ids[i * 4 + 0] != 0u;
        return true;
    }

    float maxGrassMotion(const Frame& f, int& count) {
        float m = 0.f;
        count = 0;
        for (std::size_t i = 0; i < f.grass.size(); ++i) {
            if (!f.grass[i]) continue;
            ++count;
            m = std::max(m, std::hypot(f.mx[i], f.my[i]));
        }
        return m;
    }

    float median(std::vector<float> v) {
        if (v.empty()) return 0.f;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    }

}// namespace

int main() {
    std::printf("VulkanGrassMotion_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanGrassMotion_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    Scene scene;
    scene.add(DirectionalLight::create(Color(0xffffff), 2.f));
    scene.add(AmbientLight::create(Color(0xffffff), 0.5f));
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x4a7a30);
    material->side = Side::Double;
    auto grass = GrassMesh::create(makeBlades(), material);
    grass->params.windStrength = kWind;
    grass->params.windDir = Vector2(1.f, 0.f);
    scene.add(grass);

    PerspectiveCamera camera(kFovDeg, float(kW) / float(kH), 0.1f, 100.f);
    camera.position.set(0.f, kCamY, kCamDist);
    camera.lookAt(0.f, kCamY, 0.f);

    // Moments of the gust: two with a large offset from the rest pose (what the
    // stale previous pose used to report), and a 0.1 s step with a large sway
    // that STARTS far from the rest pose too. The largest step crosses the rest
    // pose, where "offset from rest" and "motion since last frame" coincide,
    // and the old motion vectors passed there.
    float tFrozenA = 0.f, tFrozenB = 0.f, tStep = 0.f, bestA = 0.f, bestB = 0.f, bestStep = 0.f;
    for (int k = 0; k < 400; ++k) {
        const float t = 0.01f * float(k);
        const float g = gust(t, 0.f, 0.f);
        if (g > bestA) bestA = g, tFrozenA = t;
        if (-g > bestB) bestB = -g, tFrozenB = t;
        const float d = std::abs(gust(t, 0.f, 0.f) - gust(t + 0.1f, 0.f, 0.f));
        if (std::abs(g) > 0.45f && d > bestStep) bestStep = d, tStep = t;
    }
    std::printf("  gust %+.2f at t = %.2f, %+.2f at t = %.2f; step %.2f from t = %.2f\n", bestA, tFrozenA, -bestB,
                tFrozenB, bestStep, tStep);

    // One frame through the canvas' frame model (a bare render() outside it is
    // submitted with the next one, so the AOVs would trail by a frame). The
    // scene clock steps with the wind clock, as a headless film sets it.
    double clock = 0.0;
    const auto renderAt = [&](float t, int frames) {
        grass->params.time = t;
        for (int i = 0; i < frames; ++i) {
            clock += 1.0 / 60.0;
            renderer.setSimTime(clock);
            vk.canvas->animateOnce([&] { renderer.render(scene, camera); });
        }
    };

    renderAt(0.f, 4);// build + prime

    // ── 1. Time frozen ──────────────────────────────────────────────────────
    std::printf("\n1. time frozen\n");
    for (float t : {tFrozenA, tFrozenB}) {
        renderAt(t, 4);
        Frame f;
        check(readFrame(renderer, f), "motion and id AOVs read back");
        int n = 0;
        const float m = maxGrassMotion(f, n);
        std::printf("  t = %.2f: %d grass pixels, largest motion %.5f NDC (%.2f px)\n", t, n, m,
                    m * 0.5f * float(f.w));
        check(n > 200, "the blades cover the frame's centre");
        check(m < kZero, "a field frozen in time reports no motion");
    }

    // ── 2. One step ─────────────────────────────────────────────────────────
    std::printf("\n2. one 0.1 s step\n");
    renderAt(tStep, 3);
    renderAt(tStep + 0.1f, 1);
    {
        Frame f;
        check(readFrame(renderer, f), "motion and id AOVs read back");
        int top = f.h, bottom = -1;
        for (int y = 0; y < f.h; ++y)
            for (int x = 0; x < f.w; ++x)
                if (f.grass[std::size_t(y) * f.w + x]) top = std::min(top, y), bottom = std::max(bottom, y);
        // The top fifth of the blades: hf 0.8 to 1.0. Between the vertex rows
        // at hf 0.75 and 1.0 the shown displacement is linear, so its mean over
        // that band is the value at hf 0.9 of the line through hf^2 at 0.75 and 1.
        const int bandEnd = top + (bottom - top) / 5;
        std::vector<float> gx, gy, bg;
        for (int y = 0; y < f.h; ++y) {
            for (int x = 0; x < f.w; ++x) {
                const std::size_t i = std::size_t(y) * f.w + x;
                if (f.grass[i]) {
                    if (y >= top && y <= bandEnd) gx.push_back(f.mx[i]), gy.push_back(f.my[i]);
                } else {
                    bg.push_back(std::hypot(f.mx[i], f.my[i]));
                }
            }
        }
        const float hfBand = 0.5625f + 0.6f * 0.4375f;
        const float ndcPerMetre = 1.f / (kCamDist * std::tan(kFovDeg * 0.5f * 3.14159265f / 180.f) *
                                         (float(kW) / float(kH)));
        std::vector<float> expect;
        for (float x : bladeXs())
            expect.push_back(kWind * hfBand * (gust(tStep, x, 0.f) - gust(tStep + 0.1f, x, 0.f)) * ndcPerMetre);
        const float want = median(expect), got = median(gx), gotY = median(gy);
        const float bgMax = bg.empty() ? 0.f : *std::max_element(bg.begin(), bg.end());
        std::printf("  rows %d to %d (blade top at %d): motion x median %.5f NDC, analytic %.5f; y median %.5f; "
                    "background largest %.5f\n",
                    top, bandEnd, top, got, want, gotY, bgMax);
        check(!gx.empty(), "the blades' tops are on screen");
        check(std::abs(want) > 4 * kZero, "the step sways the blades measurably");
        check(got * want > 0.f, "the blades' motion points the way the wind moved them");
        // float16 motion and the band's spread of heights: 1.6 % off when written
        check(std::abs(got - want) < 0.10f * std::abs(want), "its size is the wind's step (within 10 %)");
        check(std::abs(gotY) < kZero, "the sway has no vertical motion");
        check(bgMax < kZero, "the empty background reports no motion");
    }

    // ── 3. The distance gate freezes the field after it swayed ──────────────
    std::printf("\n3. distance-gated freeze\n");
    renderAt(tStep + 0.2f, 1);// one more swaying frame: prev and shown pose differ
    grass->params.maxAnimDistance = 0.01f;// the camera is 3 m away: frozen
    renderAt(tStep + 0.3f, 2);
    {
        Frame f;
        check(readFrame(renderer, f), "motion and id AOVs read back");
        int n = 0;
        const float m = maxGrassMotion(f, n);
        std::printf("  %d grass pixels, largest motion %.5f NDC (%.2f px)\n", n, m, m * 0.5f * float(f.w));
        check(m < kZero, "a field the distance gate froze reports no motion");
    }

    // ── 4. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the grass motion path provokes no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
