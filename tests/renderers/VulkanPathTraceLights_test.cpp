// VulkanPathTraceLights_test: the path tracer's light, counted once and in
// full. Each check traces the same scene with the fix's switch on (the
// default) and off (the old way), prints both, and asserts the fixed one:
//
//   [1] an emitter through a thin pane lights the floor once, dimmed by the
//       pane (THREEPP_PT_THIN_GLASS_NEE): a clear pane takes about 8 % off,
//       a smoked one (albedo 0.2) leaves about a fifth; the old way counted
//       the light about twice;
//   [2] the last vertex's emitter NEE has its bounce (THREEPP_PT_LAST_BOUNCE_MIS):
//       a glossy floor's reflection of a panel at max_bounces 0 is the one
//       max_bounces 1 draws; the old way lost all but the NEE's MIS share;
//
// THREEPP_TEST_IMAGES=<dir>: the traces are also written there as PNGs.
// Exit 42 (SKIP) without a Vulkan device with ray query.

#include "threepp/threepp.hpp"

#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"

#include "vulkan_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 200, kH = 150;

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

    std::shared_ptr<Texture> constantEnv(float r, float g, float b) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 1.f);
        for (size_t i = 0; i < data.size(); i += 4) {
            data[i]     = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex        = Texture::create(img);
        tex->format     = Format::RGBA;
        tex->type       = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping    = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    double lum(const std::vector<float>& px, int x0, int x1, int y0, int y1) {
        double s = 0;
        int    n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const float* p = &px[(std::size_t(y) * kW + x) * 3];
                s += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
                ++n;
            }
        return n ? s / n : 0.0;
    }

    struct Tracer {
        VulkanRenderer& renderer;

        std::vector<float> trace(Scene& scene, Camera& camera, uint32_t spp, uint32_t maxBounces,
                                 const std::string& name) {
            VulkanRenderer::PathTraceParams p;
            p.samplesPerFrame = 16;
            p.maxBounces      = maxBounces;
            for (int i = 0; i < 2; ++i) renderer.render(scene, camera);
            std::vector<float> rgb;
            int w = 0, h = 0;
            if (!renderer.pathTrace(scene, camera, spp, rgb, w, h, p) || w != kW || h != kH) {
                check(false, "a trace of " + name);
                return std::vector<float>(std::size_t(kW) * kH * 3, 0.f);
            }
            if (const char* dir = std::getenv("THREEPP_TEST_IMAGES"); dir && *dir)
                renderer.pathTraceToFile(scene, camera, std::filesystem::path(dir) / (name + ".png"), spp, p);
            return rgb;
        }
    };

    std::shared_ptr<Mesh> floorPlane(float albedo, float roughness, float metalness = 0.f) {
        auto mat       = MeshStandardMaterial::create();
        mat->color     = Color(albedo, albedo, albedo);
        mat->roughness = roughness;
        mat->metalness = metalness;
        auto floor        = Mesh::create(PlaneGeometry::create(8.f, 8.f), mat);
        floor->rotation.x = -math::PI / 2;
        return floor;
    }

    // An emissive square of side `size` at height y, its front face down.
    std::shared_ptr<Mesh> panel(float size, float y, float intensity, Side side) {
        auto mat               = MeshStandardMaterial::create();
        mat->color             = Color(0.f, 0.f, 0.f);
        mat->emissive          = Color(1.f, 1.f, 1.f);
        mat->emissiveIntensity = intensity;
        mat->side              = side;
        auto m        = Mesh::create(PlaneGeometry::create(size, size), mat);
        m->rotation.x = math::PI / 2;// +z (the front) turned to -y
        m->position.y = y;
        return m;
    }

}// namespace

int main() {
    std::printf("VulkanPathTraceLights_test\n");
    setEnv("THREEPP_VULKAN_VALIDATION", "1");

    auto vk = vktest::openHeadlessVulkan("VulkanPathTraceLights_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    if (!renderer.rayTracingSupported()) {
        std::printf("[skip] no ray query\n");
        return vktest::kSkipCode;
    }
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setAutoLod(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping         = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    Tracer tr{renderer};
    auto   black = constantEnv(0.f, 0.f, 0.f);

    // ── 1. An emitter through a thin pane ──────────────────────────────────
    // A small panel 2 m over a white floor, a thin pane 1 m over it between
    // them; the camera looks at the floor under the panel from below the pane.
    std::printf("\n[1] an emitter through a thin pane lights the floor once\n");
    {
        Scene scene;
        scene.environment = black;
        scene.background  = black;
        scene.add(floorPlane(0.8f, 1.f));
        scene.add(panel(0.4f, 2.f, 30.f, Side::Double));
        auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0, 0.6f, 3.5f);
        camera->lookAt(0, 0, 0);
        camera->updateMatrixWorld();

        const auto floorLum = [&](const std::string& name) {
            const auto img = tr.trace(scene, *camera, 256, 2, name);
            return lum(img, kW / 2 - 12, kW / 2 + 12, kH / 2 - 6, kH / 2 + 6);
        };
        const double open = floorLum("1_open");

        auto glass          = MeshPhysicalMaterial::create();
        glass->transmission = 1.f;
        glass->roughness    = 0.f;
        glass->metalness    = 0.f;
        glass->ior          = 1.5f;
        glass->thickness    = 0.f;
        glass->side         = Side::Double;
        auto pane           = Mesh::create(PlaneGeometry::create(3.f, 3.f), glass);
        pane->rotation.x    = -math::PI / 2;
        pane->position.y    = 1.f;
        scene.add(pane);

        const double clearOn = floorLum("1_clear_on");
        setEnv("THREEPP_PT_THIN_GLASS_NEE", "0");
        const double clearOff = floorLum("1_clear_off");
        setEnv("THREEPP_PT_THIN_GLASS_NEE", "1");

        glass->color = Color(0.2f, 0.2f, 0.2f);
        glass->needsUpdate();
        const double smokedOn = floorLum("1_smoked_on");
        setEnv("THREEPP_PT_THIN_GLASS_NEE", "0");
        const double smokedOff = floorLum("1_smoked_off");
        setEnv("THREEPP_PT_THIN_GLASS_NEE", "1");

        std::printf("  floor: no pane %.4f | clear pane %.4f (old %.4f) | smoked pane %.4f (old %.4f)\n",
                    open, clearOn, clearOff, smokedOn, smokedOff);
        std::printf("  over the open floor: clear %.3f (old %.3f), smoked %.3f (old %.3f)\n",
                    clearOn / open, clearOff / open, smokedOn / open, smokedOff / open);
        // Two faces of 4 % Fresnel each: 0.92; smoked, 0.2 of that.
        check(open > 0.05, "the panel lights the open floor");
        check(std::abs(clearOn / open - 0.92) < 0.06, "a clear pane lets 0.92 through, once");
        check(std::abs(smokedOn / open - 0.184) < 0.03, "a smoked pane (0.2) lets 0.18 through");
    }

    // ── 2. The last vertex's bounce ────────────────────────────────────────
    // A polished metal floor (roughness 0.1) under a large panel, the camera
    // on the panel's reflection. With no bounce after the floor (max_bounces
    // 0) the reflection is the floor's direct light, all of it: what one more
    // bounce sees of the panel (max_bounces 1, the panel black).
    std::printf("\n[2] the last vertex's emitter NEE has its bounce\n");
    {
        Scene scene;
        scene.environment = black;
        scene.background  = black;
        auto floor = floorPlane(0.9f, 0.1f, 1.f);
        scene.add(floor);
        scene.add(panel(2.f, 1.5f, 0.8f, Side::Double));
        auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0, 0.8f, 3.f);
        camera->lookAt(0, 0, 1.9f);
        camera->updateMatrixWorld();
        const auto refl = [&](uint32_t bounces, const std::string& name) {
            const auto img = tr.trace(scene, *camera, 256, bounces, name);
            return lum(img, kW / 2 - 15, kW / 2 + 15, kH / 2 - 8, kH / 2 + 8);
        };
        const double full = refl(1, "2_bounces1");
        const double last = refl(0, "2_bounces0_on");
        setEnv("THREEPP_PT_LAST_BOUNCE_MIS", "0");
        const double old = refl(0, "2_bounces0_off");
        setEnv("THREEPP_PT_LAST_BOUNCE_MIS", "1");
        std::printf("  reflection: max_bounces 1 %.4f | max_bounces 0 %.4f (old %.4f)\n", full, last, old);
        check(full > 0.1, "the floor mirrors the panel");
        check(std::abs(last / full - 1.0) < 0.1, "max_bounces 0 keeps the whole reflection");
    }

    // ── Validation ─────────────────────────────────────────────────────────
    if (vulkan::validationActive()) {
        const std::uint32_t errs = vulkan::validationErrorCount() - errorsBefore;
        std::printf("\n[info] validation errors: %u\n", errs);
        check(errs == 0, "no validation errors");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
