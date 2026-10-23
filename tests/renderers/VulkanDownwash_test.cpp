// VulkanDownwash_test — the rotor downwash's renderer paths (scenes/Downwash.hpp,
// VulkanRenderer::setDownwash).
//
// A flat water plane (MeshStandardMaterial::ripples) seen from straight above,
// and a ring of grass blades round a hovering jet seen from the side. Read back from the G-buffer
// AOVs (Normal: world normal + roughness; Ids: which object; Motion):
//
//   1. Without a downwash, and with one on a plane whose ripples are 0, the
//      water's normal is the plane's: the feature costs a scene without it
//      nothing it can see.
//   2. With ripples = 1 the water's normal tilts along the radius from the
//      jet's axis (concentric rings), and the struck disc is rougher.
//   3. The grass leans out: seen from the side, its pixels spread away from
//      the axis, although the field's distance gate (maxAnimDistance) has
//      frozen it. With both clocks still, its motion vectors stay zero.
//   4. The jet moves: the grass reports motion. The jet goes: the field,
//      still gated, stands up again to the spread it had before.
//   5. No validation errors.
//
// Plain exit-code program, like the other renderer tests here. Exits 42 (CTest
// "Skipped") when no Vulkan/RT GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Downwash.hpp"

#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 320, kH = 240;
    constexpr std::uint32_t kGrassId = 7, kWaterId = 9;

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

    // A ring of blades round the axis, each a strip across its radius.
    std::shared_ptr<BufferGeometry> makeBladeRing() {
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        std::vector<float> pos, nrm, hf;
        std::vector<unsigned int> idx;
        constexpr float kH = 0.35f, kHalfW = 0.03f;
        for (int b = 0; b < 240; ++b) {
            const float a = 6.2831853f * u01(rng);
            const float r = std::sqrt(0.16f + u01(rng) * (9.f - 0.16f));
            const float x = r * std::cos(a), z = r * std::sin(a);
            const float tx = -std::sin(a) * kHalfW, tz = std::cos(a) * kHalfW;
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            for (int s = 0; s <= 4; ++s) {
                const float f = float(s) / 4.f;
                pos.insert(pos.end(), {x - tx, f * kH, z - tz, x + tx, f * kH, z + tz});
                nrm.insert(nrm.end(), {std::cos(a), 0.f, std::sin(a), std::cos(a), 0.f, std::sin(a)});
                hf.insert(hf.end(), {f, f});
            }
            for (unsigned int s = 0; s < 4u; ++s) {
                const unsigned int i0 = base + 2 * s;
                idx.insert(idx.end(), {i0, i0 + 1, i0 + 3, i0, i0 + 3, i0 + 2});
            }
        }
        auto g = BufferGeometry::create();
        g->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        g->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        g->setAttribute("heightFrac", FloatBufferAttribute::create(hf, 1));
        g->setIndex(idx);
        return g;
    }

    struct Frame {
        int w = 0, h = 0;
        std::vector<float> nx, ny, nz, rough, mx, my;
        std::vector<std::uint32_t> id;
    };

    bool readFrame(VulkanRenderer& r, Frame& f) {
        std::vector<std::uint8_t> raw;
        int w = 0, h = 0, bpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Normal, raw, w, h, bpp) || bpp != 8) return false;
        f.w = w;
        f.h = h;
        const std::size_t n = std::size_t(w) * std::size_t(h);
        f.nx.resize(n), f.ny.resize(n), f.nz.resize(n), f.rough.resize(n);
        const auto* hn = reinterpret_cast<const std::uint16_t*>(raw.data());
        for (std::size_t i = 0; i < n; ++i) {
            f.nx[i] = halfToFloat(hn[i * 4 + 0]) * 2.f - 1.f;
            f.ny[i] = halfToFloat(hn[i * 4 + 1]) * 2.f - 1.f;
            f.nz[i] = halfToFloat(hn[i * 4 + 2]) * 2.f - 1.f;
            f.rough[i] = halfToFloat(hn[i * 4 + 3]);
        }
        int mw = 0, mh = 0, mbpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Motion, raw, mw, mh, mbpp) || mw != w || mh != h ||
            mbpp != 8)
            return false;
        f.mx.resize(n), f.my.resize(n);
        const auto* hm = reinterpret_cast<const std::uint16_t*>(raw.data());
        for (std::size_t i = 0; i < n; ++i) {
            f.mx[i] = halfToFloat(hm[i * 4 + 0]);
            f.my[i] = halfToFloat(hm[i * 4 + 1]);
        }
        int iw = 0, ih = 0, ibpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, iw, ih, ibpp) || iw != w || ih != h || ibpp != 16)
            return false;
        const auto* ids = reinterpret_cast<const std::uint32_t*>(raw.data());
        f.id.resize(n);
        for (std::size_t i = 0; i < n; ++i) f.id[i] = ids[i * 4 + 1];
        return true;
    }

    // Seen from the side, the grass's pixels' mean horizontal distance from
    // the image's centre column (the jet's axis): blades leaning out from the
    // axis spread it.
    float grassSpread(const Frame& f, int& count) {
        double sum = 0;
        count = 0;
        for (int y = 0; y < f.h; ++y)
            for (int x = 0; x < f.w; ++x)
                if (f.id[std::size_t(y) * f.w + x] == kGrassId) {
                    sum += std::abs(x + 0.5 - f.w * 0.5);
                    ++count;
                }
        return count ? float(sum / count) : 0.f;
    }

    float maxGrassMotion(const Frame& f) {
        float m = 0.f;
        for (std::size_t i = 0; i < f.id.size(); ++i)
            if (f.id[i] == kGrassId) m = std::max(m, std::hypot(f.mx[i], f.my[i]));
        return m;
    }

    struct WaterStats {
        int n = 0;
        float maxTilt = 0.f, p90Tilt = 0.f, radialShare = 0.f, discRough = 0.f, farRough = 0.f;
    };

    // The water's pixels in a band round the axis, clear of the blades.
    WaterStats waterStats(const Frame& f) {
        WaterStats s;
        std::vector<float> tilts, radial;
        double discR = 0, farR = 0;
        int discN = 0, farN = 0;
        for (int y = 0; y < f.h; ++y) {
            for (int x = 0; x < f.w; ++x) {
                const std::size_t i = std::size_t(y) * f.w + x;
                if (f.id[i] != kWaterId) continue;
                const float dx = x + 0.5f - f.w * 0.5f, dy = y + 0.5f - f.h * 0.5f;
                const float rr = std::hypot(dx, dy) / float(f.h);
                const float t = std::hypot(f.nx[i], f.nz[i]);
                s.maxTilt = std::max(s.maxTilt, t);
                if (rr < 0.04f) discR += f.rough[i], ++discN;
                if (rr > 0.48f) farR += f.rough[i], ++farN;
                if (rr < 0.08f || rr > 0.45f) continue;
                ++s.n;
                tilts.push_back(t);
                // Image x is world +x, image y world +z (the camera looks
                // down -y with -z up the frame).
                const float ux = dx / std::hypot(dx, dy), uz = dy / std::hypot(dx, dy);
                if (t > 0.03f) radial.push_back(std::abs(f.nx[i] * ux + f.nz[i] * uz) / t);
            }
        }
        if (!tilts.empty()) {
            std::sort(tilts.begin(), tilts.end());
            s.p90Tilt = tilts[tilts.size() * 9 / 10];
        }
        if (!radial.empty()) {
            std::nth_element(radial.begin(), radial.begin() + radial.size() / 2, radial.end());
            s.radialShare = radial[radial.size() / 2];
        }
        s.discRough = discN ? float(discR / discN) : 0.f;
        s.farRough = farN ? float(farR / farN) : 0.f;
        return s;
    }

}// namespace

int main() {
    std::printf("VulkanDownwash_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanDownwash_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    renderer.setDlss(false);
    renderer.setFsr(false);
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    Scene scene;
    scene.add(DirectionalLight::create(Color(0xffffff), 2.f));
    scene.add(AmbientLight::create(Color(0xffffff), 0.5f));

    auto waterMat = MeshStandardMaterial::create();
    waterMat->color = Color(0.02f, 0.03f, 0.03f);
    waterMat->roughness = 0.04f;
    auto water = Mesh::create(PlaneGeometry::create(14, 14), waterMat);
    water->rotation.x = -3.14159265f / 2.f;
    scene.add(water);

    auto grassMat = MeshStandardMaterial::create();
    grassMat->color = Color(0.2f, 0.5f, 0.1f);
    grassMat->side = Side::Double;
    auto grass = GrassMesh::create(makeBladeRing(), grassMat);
    grass->params.windStrength = 0.f;
    grass->params.maxAnimDistance = 0.01f;// the camera is 5 m off: frozen unless a jet wakes it
    scene.add(grass);

    renderer.setObjectInstanceId(*grass, kGrassId);
    renderer.setObjectInstanceId(*water, kWaterId);

    PerspectiveCamera top(60.f, float(kW) / float(kH), 0.1f, 100.f);
    top.position.set(0.f, 5.f, 0.001f);
    top.lookAt(0.f, 0.f, 0.f);
    PerspectiveCamera side(50.f, float(kW) / float(kH), 0.1f, 100.f);
    side.position.set(0.f, 0.5f, 6.f);
    side.lookAt(0.f, 0.15f, 0.f);
    Camera* camera = &top;

    double clock = 2.0;
    const auto render = [&](int frames, bool advance) {
        for (int i = 0; i < frames; ++i) {
            if (advance) clock += 1.0 / 60.0;
            renderer.setSimTime(clock);
            vk.canvas->animateOnce([&] { renderer.render(scene, *camera); });
        }
    };

    render(4, true);
    Frame rest;
    check(readFrame(renderer, rest), "AOVs read back");
    camera = &side;
    render(4, true);
    Frame restSide;
    check(readFrame(renderer, restSide), "AOVs read back");
    int nRest = 0;
    const float rRest = grassSpread(restSide, nRest);
    camera = &top;

    // ── 1. Nothing visible without it ───────────────────────────────────────
    std::printf("\n1. no downwash, and a downwash on a plane without ripples\n");
    {
        const WaterStats s = waterStats(rest);
        std::printf("  no downwash: %d water pixels, largest tilt %.5f\n", s.n, s.maxTilt);
        check(s.n > 5000, "the water fills the frame");
        check(s.maxTilt < 2e-3f, "the plane's normal is flat");
    }
    auto dw = Downwash::create();
    auto& jet = dw->addJet(Vector3(0.f, 1.2f, 0.f));
    renderer.setDownwash(dw);
    render(3, true);
    {
        Frame f;
        check(readFrame(renderer, f), "AOVs read back");
        const WaterStats s = waterStats(f);
        std::printf("  downwash, ripples 0: largest tilt %.5f\n", s.maxTilt);
        check(s.maxTilt < 2e-3f, "a plane without ripples keeps its normal");
    }

    // ── 2. The rings ────────────────────────────────────────────────────────
    std::printf("\n2. ripples = 1\n");
    waterMat->ripples = 1.f;
    waterMat->needsUpdate();
    render(3, true);
    {
        Frame f;
        check(readFrame(renderer, f), "AOVs read back");
        const WaterStats s = waterStats(f);
        std::printf("  tilt 90th pct %.4f, largest %.4f; radial share median %.3f; roughness disc %.3f, far %.3f\n",
                    s.p90Tilt, s.maxTilt, s.radialShare, s.discRough, s.farRough);
        check(s.p90Tilt > 0.03f, "the rings tilt the water's normal");
        check(s.radialShare > 0.9f, "along the radius: the rings are concentric");
        check(s.discRough > s.farRough + 0.08f, "the struck disc is rougher than the far water");
    }

    // ── 3. The grass leans out, and holds still while both clocks do ────────
    std::printf("\n3. grass under a still jet\n");
    camera = &side;
    {
        render(3, true);
        render(2, false);
        Frame f;
        check(readFrame(renderer, f), "AOVs read back");
        int n = 0;
        const float rBent = grassSpread(f, n);
        const float m = maxGrassMotion(f);
        std::printf("  grass pixels %d -> %d, spread from the axis %.2f -> %.2f px; largest motion %.5f NDC\n",
                    nRest, n, rRest, rBent, m);
        check(nRest > 500 && n > 500, "the blades are on screen");
        check(rBent > rRest * 1.03f, "the blades lean out from the axis though the gate froze the field");
        check(m < 1e-3f, "a still jet over still grass reports no motion");
    }

    // ── 4. The jet moves, then goes ─────────────────────────────────────────
    std::printf("\n4. the jet moves, then goes\n");
    {
        jet.center.x = 0.3f;
        render(1, false);
        Frame f;
        check(readFrame(renderer, f), "AOVs read back");
        const float m = maxGrassMotion(f);
        std::printf("  jet moved 0.3 m: largest grass motion %.5f NDC\n", m);
        check(m > 2e-3f, "the bend follows the jet, and the motion vectors say so");

        dw->clear();
        render(3, true);
        Frame g;
        check(readFrame(renderer, g), "AOVs read back");
        int n = 0;
        const float rBack = grassSpread(g, n);
        std::printf("  jet gone: %d grass pixels (%d before it came), spread %.2f px (%.2f before)\n", n, nRest,
                    rBack, rRest);
        check(std::abs(rBack - rRest) < 0.01f * rRest && std::abs(n - nRest) < nRest / 50,
              "the gated field stood up again");
    }

    // ── 5. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the downwash paths provoke no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
