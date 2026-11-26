// VulkanPathTraceLights_test: the path tracer's light, counted once and in
// full. Each check traces the same scene with the fix's switch on (the
// default) and asserts it; with --old it also traces the old way (the switch
// off) and prints it beside, which no check reads:
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
//   [6] a sprite card's shadow follows its texels' alpha: the sun through an
//       alpha-masked card at opacity 0.5 is halved under its covered half and
//       whole under its clear half;
//   [7] an emissive blended card lights the floor by its opaque share: at
//       opacity 0.5, as an opaque panel of half its emission does;
//
// --old: also trace each switch's old way, as above (8 traces, about a quarter
// of the samples).
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
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 200, kH = 150;

    using vktest::check;
    using vktest::failures;
    using vktest::setEnv;
    using vktest::constantEnv;

    bool traceOldWay = false;// --old

    // The same trace with a fix's switch at "0": run only with --old, for the
    // print beside the fixed value. Empty without it.
    template<class F>
    auto oldWay(const char* knob, F&& trace) {
        using T = decltype(trace());
        if (!traceOldWay) return std::optional<T>{};
        setEnv(knob, "0");
        std::optional<T> v{trace()};
        setEnv(knob, "1");
        return v;
    }

    // " (old 0.1234)", or nothing when the old way was not traced.
    std::string oldText(const std::optional<double>& v, const char* fmt = "%.4f") {
        if (!v) return {};
        char buf[48];
        std::snprintf(buf, sizeof(buf), fmt, *v);
        return std::string(" (old ") + buf + ")";
    }

    template<class T, class F>
    std::optional<double> mapOld(const std::optional<T>& v, F&& f) {
        if (!v) return std::nullopt;
        return f(*v);
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

    // A 4x4 RGBA8 map, white, its left half (u < 0.5) opaque and its right
    // half clear, unfiltered.
    std::shared_ptr<Texture> halfMaskMap() {
        std::vector<unsigned char> data(4 * 4 * 4, 255);
        for (int y = 0; y < 4; ++y)
            for (int x = 2; x < 4; ++x) data[(y * 4 + x) * 4 + 3] = 0;
        Image img{ImageData(std::move(data)), 4u, 4u};
        auto  tex       = Texture::create(img);
        tex->format     = Format::RGBA;
        tex->magFilter  = Filter::Nearest;
        tex->minFilter  = Filter::Nearest;
        tex->generateMipmaps = false;
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

int main(int argc, char** argv) {
    std::printf("VulkanPathTraceLights_test\n");
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--old") == 0) traceOldWay = true;
    }

    auto vk = vktest::openHeadlessVulkan("VulkanPathTraceLights_test", kW, kH);
    if (!vk) return vktest::kSkipCode;
    VulkanRenderer& renderer = *vk.renderer;
    if (!renderer.rayTracingSupported()) {
        std::printf("[skip] no ray query\n");
        return vktest::kSkipCode;
    }
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();

    vktest::pinNativeOutput(renderer);
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

        const double clearOn  = floorLum("1_clear_on");
        const auto   clearOff = oldWay("THREEPP_PT_THIN_GLASS_NEE", [&] { return floorLum("1_clear_off"); });

        glass->color = Color(0.2f, 0.2f, 0.2f);
        glass->needsUpdate();
        const double smokedOn  = floorLum("1_smoked_on");
        const auto   smokedOff = oldWay("THREEPP_PT_THIN_GLASS_NEE", [&] { return floorLum("1_smoked_off"); });

        const auto perOpen = [&](double v) { return v / open; };
        std::printf("  floor: no pane %.4f | clear pane %.4f%s | smoked pane %.4f%s\n",
                    open, clearOn, oldText(clearOff).c_str(), smokedOn, oldText(smokedOff).c_str());
        std::printf("  over the open floor: clear %.3f%s, smoked %.3f%s\n",
                    clearOn / open, oldText(mapOld(clearOff, perOpen), "%.3f").c_str(),
                    smokedOn / open, oldText(mapOld(smokedOff, perOpen), "%.3f").c_str());
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
        const auto   old  = oldWay("THREEPP_PT_LAST_BOUNCE_MIS", [&] { return refl(0, "2_bounces0_off"); });
        std::printf("  reflection: max_bounces 1 %.4f | max_bounces 0 %.4f%s\n", full, last, oldText(old).c_str());
        check(full > 0.1, "the floor mirrors the panel");
        check(std::abs(last / full - 1.0) < 0.1, "max_bounces 0 keeps the whole reflection");

        // A black unlit card at opacity 0.5 between the floor and the panel
        // (above the camera's rays to the floor): the reflection through it is
        // half the open one at either depth; the last vertex's bounce stopped
        // at the card, which kept only the NEE's MIS share of the half.
        auto cardMat         = MeshBasicMaterial::create();
        cardMat->color       = Color(0.f, 0.f, 0.f);
        cardMat->map         = whiteMap();
        cardMat->transparent = true;
        cardMat->opacity     = 0.5f;
        cardMat->side        = Side::Double;
        auto card            = Mesh::create(PlaneGeometry::create(3.f, 3.f), cardMat);
        card->rotation.x     = -math::PI / 2;
        card->position.y     = 1.2f;
        scene.add(card);
        const double cardFull = refl(1, "2_card_bounces1");
        const double cardLast = refl(0, "2_card_bounces0");
        std::printf("  through a card at opacity 0.5: max_bounces 1 %.4f | max_bounces 0 %.4f (open %.4f)\n",
                    cardFull, cardLast, full);
        check(std::abs(cardFull / full - 0.5) < 0.06, "a card at opacity 0.5 halves the reflection");
        check(std::abs(cardLast / cardFull - 1.0) < 0.1, "and max_bounces 0 keeps it, through the card");
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
        // faces sideways (half the sky), the top up. Their intensities are
        // irradiance (three.js r155+, GLRenderer): pi lights a Lambert surface
        // at its albedo.
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

        auto ambient = AmbientLight::create(Color(1.f, 1.f, 1.f), math::PI);
        scene.add(ambient);
        const auto amb    = tr.trace(scene, *camera, 128, 2, "3_ambient_on");
        const auto ambOld = oldWay("THREEPP_PT_AMBIENT_LIGHTS",
                                   [&] { return tr.trace(scene, *camera, 128, 2, "3_ambient_off"); });
        scene.remove(*ambient);
        std::printf("  ambient: sphere centre %.4f%s\n", centre(amb), oldText(mapOld(ambOld, centre)).c_str());
        check(centre(amb) > 0.46 && centre(amb) < 0.60, "an ambient light of pi lights the sphere at its albedo (0.5)");

        auto hemi = HemisphereLight::create(Color(1.f, 1.f, 1.f), Color(0.f, 0.f, 0.f), math::PI);
        scene.add(hemi);
        const auto hem    = tr.trace(scene, *camera, 128, 2, "3_hemi_on");
        const auto hemOld = oldWay("THREEPP_PT_AMBIENT_LIGHTS",
                                   [&] { return tr.trace(scene, *camera, 128, 2, "3_hemi_off"); });
        scene.remove(*hemi);
        std::printf("  hemisphere: sphere top %.4f%s centre %.4f%s bottom %.4f%s\n",
                    top(hem), oldText(mapOld(hemOld, top)).c_str(),
                    centre(hem), oldText(mapOld(hemOld, centre)).c_str(),
                    bottom(hem), oldText(mapOld(hemOld, bottom)).c_str());
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
            const auto on  = tr.trace(scene, *camera, 64, 2, name + "_on");
            const auto off = oldWay("THREEPP_PT_BLEND_PASS",
                                    [&] { return tr.trace(scene, *camera, 64, 2, name + "_off"); });
            scene.remove(*card);
            const auto middle = [](const std::vector<float>& img) {
                return lum(img, kW / 2 - 10, kW / 2 + 10, kH / 2 - 10, kH / 2 + 10);
            };
            return std::pair{middle(on), mapOld(off, middle)};
        };
        const double redL = 0.2126;
        auto basic         = MeshBasicMaterial::create();
        basic->color       = Color(1.f, 0.f, 0.f);
        basic->map         = whiteMap();
        basic->transparent = true;
        basic->opacity     = 0.25f;
        const auto [bOn, bOff] = cardLum(basic, "4_unlit_blend");
        std::printf("  unlit blend, opacity 0.25: %.4f%s, expected %.4f\n", bOn, oldText(bOff).c_str(),
                    0.75 + 0.25 * redL);
        check(std::abs(bOn - (0.75 + 0.25 * redL)) < 0.03, "an unlit blended card shows the background through it");

        auto lit         = MeshStandardMaterial::create();
        lit->color       = Color(1.f, 0.f, 0.f);
        lit->roughness   = 1.f;
        lit->transparent = true;
        lit->opacity     = 0.25f;
        lit->side        = Side::Double;
        const auto [lOn, lOff] = cardLum(lit, "4_lit_blend");
        std::printf("  lit blend, opacity 0.25: %.4f%s, expected about %.4f\n", lOn, oldText(lOff).c_str(),
                    0.75 + 0.25 * redL);
        check(std::abs(lOn - (0.75 + 0.25 * redL)) < 0.05, "a lit blended card lets the background through untinted");

        auto add         = MeshBasicMaterial::create();
        add->color       = Color(0.2f, 0.4f, 0.6f);
        add->map         = whiteMap();
        add->blending    = Blending::Additive;
        add->transparent = true;
        add->opacity     = 0.5f;
        const double addL = 0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.6;
        const auto [aOn, aOff] = cardLum(add, "4_additive");
        std::printf("  additive, strength 0.5: %.4f%s, expected %.4f\n", aOn, oldText(aOff).c_str(),
                    1.0 + 0.5 * addL);
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

    // ── 6. A sprite card's shadow follows its texels' alpha ────────────────
    // A sun straight down on a white floor through a 2 m card 1 m up, black
    // and unlit at opacity 0.5, its map's left half opaque and right half
    // clear: the floor under the left half gets half the sun, under the right
    // half all of it. A shadow of the card's opacity alone halved both.
    std::printf("\n[6] a sprite card's shadow follows its texels' alpha\n");
    {
        Scene scene;
        scene.environment = black;
        scene.background  = black;
        scene.add(floorPlane(0.8f, 1.f));
        auto sun = DirectionalLight::create(Color(1.f, 1.f, 1.f), 3.f);
        sun->position.set(0, 5, 0);
        scene.add(sun);
        auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
        camera->position.set(0, 0.6f, 3.5f);
        camera->lookAt(0, 0, 0);
        camera->updateMatrixWorld();
        // The floor at x = -0.5 and x = +0.5 (about 29 px either side).
        const auto sides = [&](const std::string& name) {
            const auto img = tr.trace(scene, *camera, 64, 2, name);
            return std::pair{lum(img, kW / 2 - 37, kW / 2 - 21, kH / 2 - 6, kH / 2 + 6),
                             lum(img, kW / 2 + 21, kW / 2 + 37, kH / 2 - 6, kH / 2 + 6)};
        };
        const auto [openL, openR] = sides("6_open");

        auto mat         = MeshBasicMaterial::create();
        mat->color       = Color(0.f, 0.f, 0.f);
        mat->map         = halfMaskMap();
        mat->transparent = true;
        mat->opacity     = 0.5f;
        mat->side        = Side::Double;
        auto card        = Mesh::create(PlaneGeometry::create(2.f, 2.f), mat);
        card->rotation.x = -math::PI / 2;
        card->position.y = 1.f;
        scene.add(card);
        const auto [cardL, cardR] = sides("6_sprite");
        std::printf("  floor over the open floor: under the covered half %.3f, under the clear half %.3f\n",
                    cardL / openL, cardR / openR);
        check(openL > 0.05, "the sun lights the open floor");
        check(std::abs(cardL / openL - 0.5) < 0.05, "the card's covered half lets half the sun through");
        check(std::abs(cardR / openR - 1.0) < 0.05, "and its clear half all of it");
    }

    // ── 7. An emissive blended card lights by its opaque share ─────────────
    // A camera path adds a flat blend's emission where it does not pass the
    // card (its opacity), so the floor under an emissive card at opacity 0.5
    // is what an opaque panel of half its emission gives it.
    std::printf("\n[7] an emissive blended card lights by its opaque share\n");
    {
        const auto floorUnder = [&](float intensity, bool blended, const std::string& name) {
            Scene scene;
            scene.environment = black;
            scene.background  = black;
            scene.add(floorPlane(0.8f, 1.f));
            auto p = panel(0.4f, 2.f, intensity, Side::Double);
            if (blended) {
                const auto mat   = p->material();
                mat->transparent = true;
                mat->opacity     = 0.5f;
            }
            scene.add(p);
            auto camera = PerspectiveCamera::create(40, float(kW) / float(kH), 0.1f, 100.f);
            camera->position.set(0, 0.6f, 3.5f);
            camera->lookAt(0, 0, 0);
            camera->updateMatrixWorld();
            const auto img = tr.trace(scene, *camera, 128, 2, name);
            return lum(img, kW / 2 - 12, kW / 2 + 12, kH / 2 - 6, kH / 2 + 6);
        };
        const double opaqueHalf = floorUnder(15.f, false, "7_opaque_half");
        const double card       = floorUnder(30.f, true, "7_blended_card");
        std::printf("  floor: opaque panel of 15 %.4f | card of 30 at opacity 0.5 %.4f (ratio %.3f)\n",
                    opaqueHalf, card, card / opaqueHalf);
        check(opaqueHalf > 0.05, "the panel lights the floor");
        check(std::abs(card / opaqueHalf - 1.0) < 0.1, "a card at opacity 0.5 lights it as half its emission");
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
