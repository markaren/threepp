// VulkanPathTraceLights_test: the path tracer's light, counted once and in
// full. Each check traces the same scene with the fix's switch on (the
// default) and off (the old way), prints both, and asserts the fixed one:
//
//   [1] an emitter through a thin pane lights the floor once, dimmed by the
//       pane (THREEPP_PT_THIN_GLASS_NEE): a clear pane takes about 8 % off,
//       a smoked one (albedo 0.2) leaves about a fifth, a red one (1, 0, 0)
//       lights it red; the old way counted the light about twice;
//   [2] the last vertex's emitter NEE has its bounce (THREEPP_PT_LAST_BOUNCE_MIS):
//       a glossy floor's reflection of a panel at max_bounces 0 is the one
//       max_bounces 1 draws; the old way lost all but the NEE's MIS share;
//   [3] a RectAreaLight lights a floor once (through its emissive quad), and
//       AmbientLight and HemisphereLight light a sphere as the frame does
//       (THREEPP_PT_AMBIENT_LIGHTS); the old way left both out;
//   [4] blended and additive cards pass the ray (THREEPP_PT_BLEND_PASS): a flat
//       blend shows the background untinted by its transmission, an additive
//       card adds its glow; the old way drew unlit ones as opaque cards;
//   [5] a mirrored single-sided emitter's photons leave its front (no switch):
//       a glass ball's caustic is the same under the panel and its mirror;
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

    // A white 4x4 RGBA8 map: an untextured transparent MeshBasicMaterial is
    // drawn as a UI overlay and never traced; a textured one is.
    std::shared_ptr<Texture> whiteMap() {
        std::vector<unsigned char> data(4 * 4 * 4, 255);
        Image img{ImageData(std::move(data)), 4u, 4u};
        auto  tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->needsUpdate();
        return tex;
    }

    // The mean of channel c (0 red, 1 green, 2 blue) over the window.
    double chan(const std::vector<float>& px, int c, int x0, int x1, int y0, int y1) {
        double s = 0;
        int    n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                s += px[(std::size_t(y) * kW + x) * 3 + c];
                ++n;
            }
        return n ? s / n : 0.0;
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

        // A red pane: the open floor is lit white (its channels equal), the
        // floor under the pane is 0.92 of it in red and none in green; a
        // grey shadow channel lit it with the pane's luminance (0.21 x 0.92)
        // in every channel.
        glass->color = Color(1.f, 0.f, 0.f);
        glass->needsUpdate();
        const auto   red  = tr.trace(scene, *camera, 256, 2, "1_red_on");
        const double redR = chan(red, 0, kW / 2 - 12, kW / 2 + 12, kH / 2 - 6, kH / 2 + 6);
        const double redG = chan(red, 1, kW / 2 - 12, kW / 2 + 12, kH / 2 - 6, kH / 2 + 6);
        std::printf("  red pane over the open floor: red %.3f, green %.3f\n", redR / open, redG / open);
        check(std::abs(redR / open - 0.92) < 0.06, "a red pane lets 0.92 of the red through");
        check(redG / open < 0.05, "and none of the green");
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

    // ── 3. Rect, ambient and hemisphere lights ──────────────────────────────
    std::printf("\n[3] RectAreaLight, AmbientLight and HemisphereLight light surfaces\n");
    {
        // A 1 m RectAreaLight of 1 nit, 1 m over a white floor, facing it:
        // under its centre the floor is albedo x the rectangle's form factor.
        Scene scene;
        scene.environment = black;
        scene.background  = black;
        scene.add(floorPlane(0.8f, 1.f));
        auto rect        = RectAreaLight::create(Color(1.f, 1.f, 1.f), 1.f, 1.f, 1.f);
        rect->rotation.x = -math::PI / 2;// its emitting -z turned down
        rect->position.y = 1.f;
        scene.add(rect);
        auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0, 0.5f, 3.f);
        camera->lookAt(0, 0, 0);
        camera->updateMatrixWorld();
        // Four corner rectangles of 0.5 x 0.5 m at 1 m (the parallel-rectangle
        // form factor from a corner).
        const double A = 0.5, B = 0.5, sa = std::sqrt(1 + A * A), sb = std::sqrt(1 + B * B);
        const double ff = 4.0 * (A / sa * std::atan(B / sa) + B / sb * std::atan(A / sb)) / (2.0 * math::PI);
        const auto floorLum = [&](const std::string& name) {
            const auto img = tr.trace(scene, *camera, 256, 2, name);
            return lum(img, kW / 2 - 6, kW / 2 + 6, kH / 2 - 4, kH / 2 + 4);
        };
        // It lights the trace through its emissive quad (RectAreaLight's
        // child mesh), an emitter like any other: once.
        const double on = floorLum("3_rect");
        std::printf("  floor under the rect: %.4f, albedo x form factor %.4f\n", on, 0.8 * ff);
        check(std::abs(on / (0.8 * ff) - 1.0) < 0.1, "a RectAreaLight lights the floor once, through its quad");
    }
    {
        // A grey sphere under a white AmbientLight, then under a hemisphere
        // light of white sky and black ground, nothing else: the centre
        // faces sideways (half the sky), the top up.
        Scene scene;
        scene.environment = black;
        scene.background  = black;
        auto mat       = MeshStandardMaterial::create();
        mat->color     = Color(0.5f, 0.5f, 0.5f);
        mat->roughness = 1.f;
        scene.add(Mesh::create(SphereGeometry::create(1.f, 48, 32), mat));
        auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0, 0, 4.f);
        camera->lookAt(0, 0, 0);
        camera->updateMatrixWorld();
        const auto centre = [&](const std::vector<float>& img) { return lum(img, kW / 2 - 6, kW / 2 + 6, kH / 2 - 6, kH / 2 + 6); };
        const auto top    = [&](const std::vector<float>& img) { return lum(img, kW / 2 - 6, kW / 2 + 6, kH / 2 - 44, kH / 2 - 36); };
        const auto bottom = [&](const std::vector<float>& img) { return lum(img, kW / 2 - 6, kW / 2 + 6, kH / 2 + 36, kH / 2 + 44); };

        auto ambient = AmbientLight::create(Color(1.f, 1.f, 1.f), 1.f);
        scene.add(ambient);
        const auto amb = tr.trace(scene, *camera, 128, 2, "3_ambient_on");
        setEnv("THREEPP_PT_AMBIENT_LIGHTS", "0");
        const auto ambOld = tr.trace(scene, *camera, 128, 2, "3_ambient_off");
        setEnv("THREEPP_PT_AMBIENT_LIGHTS", "1");
        scene.remove(*ambient);
        std::printf("  ambient: sphere centre %.4f (old %.4f)\n", centre(amb), centre(ambOld));
        check(centre(amb) > 0.46 && centre(amb) < 0.60, "an ambient light of 1 lights the sphere at its albedo (0.5)");

        auto hemi = HemisphereLight::create(Color(1.f, 1.f, 1.f), Color(0.f, 0.f, 0.f), 1.f);
        scene.add(hemi);
        const auto hem = tr.trace(scene, *camera, 128, 2, "3_hemi_on");
        setEnv("THREEPP_PT_AMBIENT_LIGHTS", "0");
        const auto hemOld = tr.trace(scene, *camera, 128, 2, "3_hemi_off");
        setEnv("THREEPP_PT_AMBIENT_LIGHTS", "1");
        scene.remove(*hemi);
        std::printf("  hemisphere: sphere top %.4f centre %.4f bottom %.4f (old %.4f %.4f %.4f)\n", top(hem), centre(hem),
                    bottom(hem), top(hemOld), centre(hemOld), bottom(hemOld));
        // Lambert under the sky half: albedo x (0.5 N.up + 0.5); the top band
        // is at N.up ~ 0.78 (0.45), the centre at 0 (0.25), the bottom ~ -0.78.
        check(std::abs(centre(hem) - 0.25) < 0.04, "the hemisphere lights the sphere's side by half its sky");
        check(top(hem) > 0.38 && bottom(hem) < 0.10, "and its top by the sky, its bottom by the black ground");
    }

    // ── 4. Blended and additive cards ──────────────────────────────────────
    // A card over the background: a flat blend (opacity 0.25) shows 0.75 of
    // the background untinted and 0.25 of itself; an additive one (strength
    // 0.5) shows all of the background and half of its colour on top.
    std::printf("\n[4] blended and additive cards pass the ray\n");
    {
        Scene scene;
        auto white        = constantEnv(1.f, 1.f, 1.f);
        scene.environment = white;
        scene.background  = white;
        auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0, 0, 4.f);
        camera->lookAt(0, 0, 0);
        camera->updateMatrixWorld();
        const auto cardLum = [&](const std::shared_ptr<Material>& mat, const std::string& name) {
            auto card = Mesh::create(PlaneGeometry::create(1.5f, 1.5f), mat);
            scene.add(card);
            const auto on = tr.trace(scene, *camera, 64, 2, name + "_on");
            setEnv("THREEPP_PT_BLEND_PASS", "0");
            const auto off = tr.trace(scene, *camera, 64, 2, name + "_off");
            setEnv("THREEPP_PT_BLEND_PASS", "1");
            scene.remove(*card);
            return std::pair{lum(on, kW / 2 - 10, kW / 2 + 10, kH / 2 - 10, kH / 2 + 10),
                             lum(off, kW / 2 - 10, kW / 2 + 10, kH / 2 - 10, kH / 2 + 10)};
        };
        const double redL = 0.2126;
        auto basic         = MeshBasicMaterial::create();
        basic->color       = Color(1.f, 0.f, 0.f);
        basic->map         = whiteMap();
        basic->transparent = true;
        basic->opacity     = 0.25f;
        const auto [bOn, bOff] = cardLum(basic, "4_unlit_blend");
        std::printf("  unlit blend, opacity 0.25: %.4f (old %.4f), expected %.4f\n", bOn, bOff, 0.75 + 0.25 * redL);
        check(std::abs(bOn - (0.75 + 0.25 * redL)) < 0.03, "an unlit blended card shows the background through it");

        auto lit         = MeshStandardMaterial::create();
        lit->color       = Color(1.f, 0.f, 0.f);
        lit->roughness   = 1.f;
        lit->transparent = true;
        lit->opacity     = 0.25f;
        lit->side        = Side::Double;
        const auto [lOn, lOff] = cardLum(lit, "4_lit_blend");
        std::printf("  lit blend, opacity 0.25: %.4f (old %.4f), expected about %.4f\n", lOn, lOff, 0.75 + 0.25 * redL);
        check(std::abs(lOn - (0.75 + 0.25 * redL)) < 0.05, "a lit blended card lets the background through untinted");

        auto add         = MeshBasicMaterial::create();
        add->color       = Color(0.2f, 0.4f, 0.6f);
        add->map         = whiteMap();
        add->blending    = Blending::Additive;
        add->transparent = true;
        add->opacity     = 0.5f;
        const double addL = 0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.6;
        const auto [aOn, aOff] = cardLum(add, "4_additive");
        std::printf("  additive, strength 0.5: %.4f (old %.4f), expected %.4f\n", aOn, aOff, 1.0 + 0.5 * addL);
        check(std::abs(aOn - (1.0 + 0.5 * addL)) < 0.03, "an additive card adds its glow to the background");
    }

    // ── 5. A mirrored single-sided emitter's photons ───────────────────────
    // A front-side panel over a solid glass ball over a floor: the ball's
    // caustic on the floor under it (photons; the floor's NEE stops at the
    // solid) is the same with the panel mirrored (scale.x = -1), whose front
    // still faces down: facing is the object's.
    std::printf("\n[5] a mirrored single-sided emitter's photons leave its front\n");
    {
        const auto causticLum = [&](float sx, const std::string& name) {
            Scene scene;
            scene.environment = black;
            scene.background  = black;
            scene.add(floorPlane(0.8f, 1.f));
            auto p     = panel(0.5f, 2.f, 30.f, Side::Front);
            p->scale.x = sx;
            scene.add(p);
            auto glass          = MeshPhysicalMaterial::create();
            glass->transmission = 1.f;
            glass->roughness    = 0.f;
            glass->ior          = 1.5f;
            glass->thickness    = 0.8f;
            auto ball           = Mesh::create(SphereGeometry::create(0.4f, 48, 32), glass);
            ball->position.y    = 0.8f;
            scene.add(ball);
            auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
            camera->position.set(0, 0.3f, 3.f);
            camera->lookAt(0, 0, 0);
            camera->updateMatrixWorld();
            const auto img = tr.trace(scene, *camera, 128, 3, name);
            return lum(img, kW / 2 - 8, kW / 2 + 8, kH / 2 - 3, kH / 2 + 3);
        };
        const double plain    = causticLum(1.f, "5_panel");
        const double mirrored = causticLum(-1.f, "5_panel_mirrored");
        std::printf("  floor under the ball: panel %.4f, mirrored panel %.4f\n", plain, mirrored);
        check(plain > 0.02, "the ball focuses the panel's light on the floor");
        check(std::abs(mirrored / plain - 1.0) < 0.15, "and as much of the mirrored panel's");
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
