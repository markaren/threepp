// VulkanLodDither_test - a dithered LOD crossfade covers every pixel exactly
// once, in the shares its fades ask for, and the temporal resolve blends it.
//
// Mesh::setFade / InstancedMesh::setFadeAt draw a share `fade` of a mesh's
// pixels through a per-frame screen-space hash; the complement pattern draws
// exactly the pixels the plain one leaves out, so an outgoing level at fade f
// and an incoming one at 1 - f (complement) swap without a pop.
//
// The scene: one InstancedMesh of two coincident quads facing the camera in a
// uniform environment, instance A red with a plain fade, instance B blue with
// the complement fade, B 1 cm in front of A. Where both drew a pixel B wins
// the depth test, so in the raw Ids AOV (no temporal resolve) B's share is
// all of B's coverage, A's share is what A drew outside B, and a pixel
// neither drew shows the sky behind them. Then the order is swapped (A in
// front), which measures A's own coverage the same way.
//
//   1. Fade 1 / fade 0: fully drawn and absent, plain and complement alike,
//      and fade 0 on a lone quad leaves only the sky.
//   2. f in {0.1, 0.25, 0.5, 0.7, 0.9}, both depth orders: no pixel of the
//      quad is uncovered, the front quad's share is its fade, the back one's
//      the rest (each within 0.02).
//   3. The pattern changes from frame to frame (or the temporal resolve
//      could not blend it).
//   4. Two plain Meshes with Mesh::setFade: the same partition.
//   5. With the temporal resolve settled, the colour is the blend of the
//      two quads' colours (each fully drawn alone) within 0.04 of the fade,
//      measured in display units, where the resolve accumulates.
//   6. Shadows, ray-traced (when the GPU traces) and shadow maps: two levels
//      that do not coincide throw ONE shadow during the crossfade, the plain
//      level's while its fade is at least 0.5, else the complement's.
//   7. No validation errors.
//
// Plain exit-code program, like the other renderer tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include "vulkan_test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 320, kH = 240;
    constexpr float kQuadW = 1.6f, kQuadH = 1.2f;
    constexpr float kFront = 0.01f;// the front quad's offset toward the camera
    constexpr float kShareTol = 0.02f;
    constexpr float kBlendTol = 0.04f;

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

    std::shared_ptr<Texture> constantEnv(float le) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, le);
        for (size_t i = 3; i < data.size(); i += 4) data[i] = 1.f;
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    // The visible entry id (ids .x) of every pixel.
    std::vector<std::uint32_t> readIds(VulkanRenderer& r) {
        std::vector<std::uint8_t> raw;
        int w = 0, h = 0, bpp = 0;
        if (!r.readGBufferAOV(VulkanRenderer::GBufferAOV::Ids, raw, w, h, bpp) || w != kW || h != kH || bpp != 16)
            return {};
        std::vector<std::uint32_t> ids(std::size_t(kW) * kH);
        const auto* p = reinterpret_cast<const std::uint32_t*>(raw.data());
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = p[i * 4];
        return ids;
    }

    // Pixels whose 5x5 neighbourhood is all inside `in`: the quad's interior,
    // clear of the edge pixels the raster jitter moves in and out.
    std::vector<bool> erode(const std::vector<bool>& in) {
        std::vector<bool> out(in.size(), false);
        for (int y = 2; y < kH - 2; ++y) {
            for (int x = 2; x < kW - 2; ++x) {
                bool all = true;
                for (int dy = -2; dy <= 2 && all; ++dy)
                    for (int dx = -2; dx <= 2 && all; ++dx) all = in[std::size_t(y + dy) * kW + (x + dx)];
                out[std::size_t(y) * kW + x] = all;
            }
        }
        return out;
    }

    struct Shares {
        int n = 0, a = 0, b = 0, other = 0;
        float sa() const { return n ? float(a) / float(n) : 0.f; }
        float sb() const { return n ? float(b) / float(n) : 0.f; }
    };

    Shares count(const std::vector<std::uint32_t>& ids, const std::vector<bool>& mask, std::uint32_t idA,
                 std::uint32_t idB) {
        Shares s;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (!mask[i]) continue;
            ++s.n;
            if (ids[i] == idA) ++s.a;
            else if (ids[i] == idB) ++s.b;
            else ++s.other;
        }
        return s;
    }

    double srgbToLinear(double c) {
        c /= 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    // Mean linear (r, b) over the mask, and the per-pixel spread of r.
    // Mean red and blue over the mask in DISPLAY units (the 8-bit output / 255:
    // the temporal resolve accumulates post-tonemap, display-encoded colour,
    // so that is where a dither blends), the same in linear for the report,
    // and the per-pixel spread of the display red.
    struct Colour {
        double r = 0, b = 0, rLin = 0, bLin = 0, rSpread = 0;
    };

    Colour meanColour(const std::vector<unsigned char>& px, const std::vector<bool>& mask) {
        Colour c;
        int n = 0;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i]) continue;
            c.r += px[i * 3 + 0] / 255.0;
            c.b += px[i * 3 + 2] / 255.0;
            c.rLin += srgbToLinear(px[i * 3 + 0]);
            c.bLin += srgbToLinear(px[i * 3 + 2]);
            ++n;
        }
        if (n == 0) return c;
        c.r /= n;
        c.b /= n;
        c.rLin /= n;
        c.bLin /= n;
        double v = 0;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i]) continue;
            const double d = px[i * 3 + 0] / 255.0 - c.r;
            v += d * d;
        }
        c.rSpread = std::sqrt(v / n);
        return c;
    }

}// namespace

int main() {
    std::printf("VulkanLodDither_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanLodDither_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    const auto env = constantEnv(0.5f);
    Scene scene;
    scene.background = env;
    scene.environment = env;

    PerspectiveCamera camera(40.f, float(kW) / float(kH), 0.1f, 100.f);
    camera.position.set(0.f, 0.f, 3.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    auto material = MeshStandardMaterial::create();
    material->color = Color(1, 1, 1);
    material->roughness = 1.f;
    material->metalness = 0.f;
    auto pair = InstancedMesh::create(PlaneGeometry::create(kQuadW, kQuadH), material, 2);
    // Unsaturated: the GI history is demodulated by albedo, and a channel
    // with no albedo has no irradiance to carry between the two levels.
    pair->setColorAt(0, Color(0.9f, 0.35f, 0.35f));// A
    pair->setColorAt(1, Color(0.35f, 0.35f, 0.9f));// B
    pair->instanceColor()->needsUpdate();
    const auto place = [&](float zA, float zB) {
        Matrix4 m;
        m.makeTranslation(0.f, 0.f, zA);
        pair->setMatrixAt(0, m);
        m.makeTranslation(0.f, 0.f, zB);
        pair->setMatrixAt(1, m);
        pair->instanceMatrix()->needsUpdate();
        pair->computeBoundingSphere();
    };
    // Which of the two sits in front (kFront nearer the camera).
    const auto order = [&](bool bInFront) {
        place(bInFront ? 0.f : kFront, bInFront ? kFront : 0.f);
    };
    order(true);
    scene.add(pair);

    // One frame through the canvas' frame model (a bare render() outside it is
    // submitted with the next one, so the AOVs would trail by a frame).
    double clock = 0.0;
    const auto frames = [&](int n) {
        for (int i = 0; i < n; ++i) {
            clock += 1.0 / 60.0;
            renderer.setSimTime(clock);
            vk.canvas->animateOnce([&] { renderer.render(scene, camera); });
        }
    };
    const auto fades = [&](float fa, float fb) {
        pair->setFadeAt(0, fa, false);
        pair->setFadeAt(1, fb, true);
    };

    // ── 1. Fully drawn and absent ───────────────────────────────────────────
    std::printf("\n1. fade 1 and fade 0\n");
    fades(1.f, 0.f);
    frames(4);
    const auto idsA = readIds(renderer);
    check(!idsA.empty(), "the ids AOV reads back");
    if (idsA.empty()) return 1;
    // The quad's footprint and its two ids, from frames where one is alone.
    std::vector<bool> foot(idsA.size());
    std::uint32_t idA = 0;
    for (std::size_t i = 0; i < idsA.size(); ++i) {
        foot[i] = idsA[i] != 0u;
        if (foot[i]) idA = idsA[i];
    }
    const auto mask = erode(foot);
    fades(0.f, 1.f);
    frames(2);
    const auto idsB = readIds(renderer);
    std::uint32_t idB = 0;
    for (std::size_t i = 0; i < idsB.size(); ++i)
        if (mask[i] && idsB[i] != 0u) idB = idsB[i];
    {
        const Shares sA = count(idsA, mask, idA, idB);
        const Shares sB = count(idsB, mask, idA, idB);
        std::printf("  quad interior %d px; ids A = %u, B = %u\n", sA.n, idA, idB);
        check(sA.n > 20000, "the quad covers the frame's centre");
        check(idA != 0u && idB != 0u && idA != idB, "the two instances have their own ids");
        check(sA.a == sA.n, "plain fade 1 in front of complement fade 0: all A");
        check(sB.b == sB.n, "plain fade 0 behind complement fade 1: all B");
    }
    order(false);// A in front: absent in front must not hide what is behind
    fades(0.f, 1.f);
    frames(2);
    check(count(readIds(renderer), mask, idA, idB).b == count(idsB, mask, idA, idB).n,
          "plain fade 0 in front of complement fade 1: all B");
    fades(1.f, 0.f);
    frames(2);
    check(count(readIds(renderer), mask, idA, idB).a == count(idsA, mask, idA, idB).n,
          "plain fade 1 in front of complement fade 0: all A");
    fades(0.f, 0.f);
    frames(2);
    {
        const auto ids = readIds(renderer);
        int covered = 0;
        for (std::size_t i = 0; i < ids.size(); ++i) covered += foot[i] && ids[i] != 0u ? 1 : 0;
        check(covered == 0, "both at fade 0: only the sky (" + std::to_string(covered) + " px drawn)");
    }

    // ── 2. Shares and holes ─────────────────────────────────────────────────
    std::printf("\n2. crossfade shares (A plain f, B complement 1 - f)\n");
    for (bool bInFront : {true, false}) {
        order(bInFront);
        for (float f : {0.1f, 0.25f, 0.5f, 0.7f, 0.9f}) {
            fades(f, 1.f - f);
            frames(2);
            const Shares s = count(readIds(renderer), mask, idA, idB);
            // The front quad's share is its whole coverage.
            const float front = bInFront ? s.sb() : s.sa();
            const float wantFront = bInFront ? 1.f - f : f;
            char line[160];
            std::snprintf(line, sizeof line, "%s in front, f = %.2f: A %.4f, B %.4f, uncovered %d of %d px",
                          bInFront ? "B" : "A", f, s.sa(), s.sb(), s.other, s.n);
            std::printf("  %s\n", line);
            check(s.other == 0, std::string("no hole: ") + line);
            check(std::abs(front - wantFront) < kShareTol, "the front quad draws its fade's share");
        }
    }

    // ── 3. The pattern moves ────────────────────────────────────────────────
    std::printf("\n3. per-frame pattern\n");
    order(true);
    fades(0.5f, 0.5f);
    frames(2);
    {
        const auto p0 = readIds(renderer);
        frames(1);
        const auto p1 = readIds(renderer);
        int n = 0, flipped = 0;
        for (std::size_t i = 0; i < p0.size(); ++i) {
            if (!mask[i]) continue;
            ++n;
            flipped += p0[i] != p1[i] ? 1 : 0;
        }
        const float share = n ? float(flipped) / float(n) : 0.f;
        std::printf("  %.3f of the quad's pixels changed level between two frames (independent: 0.5)\n", share);
        check(share > 0.35f && share < 0.65f, "the dither is redrawn every frame");
    }

    // ── 4. Plain meshes ─────────────────────────────────────────────────────
    std::printf("\n4. two plain Meshes with Mesh::setFade\n");
    {
        scene.remove(*pair);
        auto matA = MeshStandardMaterial::create();
        matA->color = Color(1, 0, 0);
        auto matB = MeshStandardMaterial::create();
        matB->color = Color(0, 0, 1);
        auto quadA = Mesh::create(PlaneGeometry::create(kQuadW, kQuadH), matA);
        auto quadB = Mesh::create(PlaneGeometry::create(kQuadW, kQuadH), matB);
        quadB->position.z = kFront;
        scene.add(quadA);
        scene.add(quadB);
        quadA->setFade(1.f);
        quadB->setFade(0.f, true);
        frames(3);
        std::uint32_t pa = 0, pb = 0;
        {
            const auto ids = readIds(renderer);
            for (std::size_t i = 0; i < ids.size(); ++i)
                if (mask[i] && ids[i] != 0u) pa = ids[i];
        }
        quadA->setFade(0.f);
        quadB->setFade(1.f, true);
        frames(2);
        {
            const auto ids = readIds(renderer);
            for (std::size_t i = 0; i < ids.size(); ++i)
                if (mask[i] && ids[i] != 0u) pb = ids[i];
        }
        check(pa != 0u && pb != 0u && pa != pb, "each mesh alone covers the quad");
        for (float f : {0.3f, 0.8f}) {
            quadA->setFade(f);
            quadB->setFade(1.f - f, true);
            frames(2);
            const Shares s = count(readIds(renderer), mask, pa, pb);
            std::printf("  f = %.2f: A %.4f, B %.4f, uncovered %d of %d px\n", f, s.sa(), s.sb(), s.other, s.n);
            check(s.other == 0 && std::abs(s.sb() - (1.f - f)) < kShareTol,
                  "plain meshes partition the quad in the fades' shares");
        }
        scene.remove(*quadA);
        scene.remove(*quadB);
        scene.add(pair);
    }

    // ── 5. The temporal resolve blends ──────────────────────────────────────
    std::printf("\n5. settled colour\n");
    // Coincident, as two levels of one object nearly are: with one in front,
    // the level the rays see would occlude the sky of the one behind it.
    place(0.f, 0.f);
    const auto settled = [&](float fa, float fb) {
        fades(fa, fb);
        frames(48);
        return meanColour(renderer.readRGBPixels(), mask);
    };
    const Colour pureA = settled(1.f, 0.f);
    const Colour pureB = settled(0.f, 1.f);
    std::printf("  A alone: r %.4f b %.4f; B alone: r %.4f b %.4f (display)\n", pureA.r, pureA.b, pureB.r, pureB.b);
    check(pureA.r > pureB.r + 0.1 && pureB.b > pureA.b + 0.1, "each quad alone shows its colour");
    for (float f : {0.25f, 0.5f, 0.75f}) {
        const Colour c = settled(f, 1.f - f);
        // The share of A in the blend, read off each channel.
        const double fr = (c.r - pureB.r) / std::max(pureA.r - pureB.r, 1e-6);
        const double fb = (pureB.b - c.b) / std::max(pureB.b - pureA.b, 1e-6);
        // The same in linear light: a display-space blend reads darker there
        // (the encoding is concave), which is the resolve's, not the dither's.
        const double frLin = (c.rLin - pureB.rLin) / std::max(pureA.rLin - pureB.rLin, 1e-6);
        // One frame of the dither alone spreads a pixel's red by
        // sqrt(f (1 - f)) of the difference; the resolve averages it down.
        const double rawSpread = std::sqrt(double(f) * (1.0 - f)) * std::abs(pureA.r - pureB.r);
        std::printf("  f = %.2f: share of A from red %.4f, from blue %.4f (linear red %.4f); per-pixel red "
                    "spread %.4f (one frame of dither: %.4f)\n",
                    f, fr, fb, frLin, c.rSpread, rawSpread);
        check(std::abs(fr - f) < kBlendTol && std::abs(fb - f) < kBlendTol, "the settled colour is the blend");
        check(c.rSpread < 0.5 * rawSpread, "the temporal resolve averages the dither down");
    }

    // ── 6. Shadows: one level casts ─────────────────────────────────────────
    // Two levels that do not coincide, so their shadows can be told apart: A
    // up left, B down right, a white wall 0.5 m behind them and a sun from
    // the upper left front that throws each shadow down and right onto it.
    std::printf("\n6. shadows\n");
    {
        scene.remove(*pair);
        auto sun = DirectionalLight::create(Color(1, 1, 1), 3.f);
        sun->position.set(-1.f, 1.f, 2.f);
        scene.add(sun);
        auto wallMat = MeshStandardMaterial::create();
        wallMat->color = Color(1, 1, 1);
        wallMat->roughness = 1.f;
        auto wall = Mesh::create(PlaneGeometry::create(6.f, 4.f), wallMat);
        wall->position.z = -0.5f;
        scene.add(wall);
        auto casterMat = MeshStandardMaterial::create();
        casterMat->color = Color(0.5f, 0.5f, 0.5f);
        casterMat->side = Side::Double;
        auto la = Mesh::create(PlaneGeometry::create(0.5f, 0.5f), casterMat);
        la->position.set(-0.5f, 0.25f, 0.f);
        auto lb = Mesh::create(PlaneGeometry::create(0.5f, 0.5f), casterMat);
        lb->position.set(0.5f, -0.25f, 0.f);
        scene.add(la);
        scene.add(lb);

        const auto luma = [](const std::vector<unsigned char>& px, std::size_t i) {
            return 0.2126 * px[i * 3] + 0.7152 * px[i * 3 + 1] + 0.0722 * px[i * 3 + 2];
        };
        std::vector<VulkanRenderer::ShadowMode> modes;
        if (renderer.rayTracingSupported()) modes.push_back(VulkanRenderer::ShadowMode::RayTraced);
        modes.push_back(VulkanRenderer::ShadowMode::ShadowMaps);
        for (const auto mode : modes) {
            const char* name = mode == VulkanRenderer::ShadowMode::RayTraced ? "rays" : "maps";
            renderer.setShadowMode(mode);
            const auto shot = [&](float fa, float fb) {
                la->setFade(fa);
                lb->setFade(fb, true);
                frames(40);
                return renderer.readRGBPixels();
            };
            const auto onlyA = shot(1.f, 0.f);
            const auto idsOnlyA = readIds(renderer);
            const auto onlyB = shot(0.f, 1.f);
            const auto idsOnlyB = readIds(renderer);
            const std::uint32_t wallId = idsOnlyB.empty() ? 0u : idsOnlyB[std::size_t(5) * kW + 5];
            // Each level's own shadow: pixels that show the wall with either
            // level alone (so in front of neither quad) and are clearly darker
            // with that level alone than with the other alone.
            constexpr double kStep = 30.0;// 8-bit luma
            std::vector<bool> shA(std::size_t(kW) * kH), shB(std::size_t(kW) * kH);
            int nA = 0, nB = 0;
            for (std::size_t i = 0; i < shA.size(); ++i) {
                if (idsOnlyA.empty() || idsOnlyB.empty() || idsOnlyA[i] != wallId || idsOnlyB[i] != wallId) continue;
                shA[i] = luma(onlyB, i) - luma(onlyA, i) > kStep;
                shB[i] = luma(onlyA, i) - luma(onlyB, i) > kStep;
            }
            shA = erode(shA);
            shB = erode(shB);
            for (std::size_t i = 0; i < shA.size(); ++i) nA += shA[i] ? 1 : 0, nB += shB[i] ? 1 : 0;
            check(nA > 300 && nB > 300, std::string(name) + ": each level alone throws its own shadow (" +
                                                std::to_string(nA) + " and " + std::to_string(nB) + " px)");
            // A shadow pixel is dark when it is nearer its shadowed value than
            // its lit one.
            const auto dark = [&](const std::vector<unsigned char>& img, std::size_t i, bool ofA) {
                const double v = luma(img, i), shadowed = luma(ofA ? onlyA : onlyB, i),
                             litV = luma(ofA ? onlyB : onlyA, i);
                return std::abs(v - shadowed) < std::abs(v - litV);
            };
            for (float f : {0.3f, 0.5f, 0.7f}) {
                const auto img = shot(f, 1.f - f);
                int darkInA = 0, darkInB = 0;
                for (std::size_t i = 0; i < shA.size(); ++i) {
                    if (shA[i] && dark(img, i, true)) ++darkInA;
                    if (shB[i] && dark(img, i, false)) ++darkInB;
                }
                const double sa = nA ? double(darkInA) / nA : 0, sb = nB ? double(darkInB) / nB : 0;
                const bool aCasts = f >= 0.5f;
                std::printf("  %s, f = %.2f: A's shadow %.3f dark, B's %.3f dark\n", name, f, sa, sb);
                check(aCasts ? (sa > 0.95 && sb < 0.05) : (sb > 0.95 && sa < 0.05),
                      std::string(name) + ": one shadow, the " + (aCasts ? "plain" : "complement") +
                              " level's (no double, no missing shadow)");
            }
        }
        renderer.setShadowMode(VulkanRenderer::ShadowMode::Auto);
    }

    // ── 7. Validation ───────────────────────────────────────────────────────
    if (!vulkan::validationActive()) {
        std::printf("\n[note] VK_LAYER_KHRONOS_validation is not installed: the zero-error\n"
                    "       check is skipped.\n");
    } else {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "the LOD dither path provokes no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
