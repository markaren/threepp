// Shadow-map runtime reconfiguration.
//
// Renderer::shadowMap() is mutable at runtime (the ImGui RendererSettings panel
// exposes it), and both `enabled` and `type` feed compile-time defines
// (USE_SHADOWMAP, SHADOWMAP_TYPE_*) plus the render-target layout. Changing them
// after the first frame therefore has to invalidate material programs and, for
// VSM, reallocate the shadow targets.

#include "gl_test_helpers.hpp"

#include <cstdlib>

#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"

namespace {

    // A lit plane with a box floating above it: the box casts a shadow onto the
    // plane, so "how dark is the darkest region" distinguishes shadowed from
    // unshadowed rendering.
    struct ShadowScene {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
    };

    ShadowScene makeShadowScene() {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto light = DirectionalLight::create(0xffffff, 3.f);
        light->position.set(2, 6, 2);
        light->castShadow = true;
        scene->add(light);

        // A little ambient so the unshadowed floor is clearly brighter than the
        // shadowed part rather than both being near-black.
        scene->add(AmbientLight::create(0x202020));

        auto floorMat = MeshStandardMaterial::create();
        floorMat->color = Color(1, 1, 1);
        floorMat->roughness = 1.f;
        floorMat->metalness = 0.f;
        auto floor = Mesh::create(PlaneGeometry::create(20, 20), floorMat);
        floor->rotation.x = -math::PI / 2.f;
        floor->receiveShadow = true;
        scene->add(floor);

        auto boxMat = MeshStandardMaterial::create();
        boxMat->color = Color(1, 1, 1);
        auto box = Mesh::create(BoxGeometry::create(2, 2, 2), boxMat);
        box->position.set(0, 2, 0);
        box->castShadow = true;
        scene->add(box);

        auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 100.f);
        camera->position.set(0, 7, 9);
        camera->lookAt(Vector3{0, 0, 0});

        return {scene, camera};
    }

    std::vector<unsigned char> renderOnce(GLRenderer& renderer, ShadowScene& s) {
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*s.scene, *s.camera);
        return renderer.readRGBPixels();
    }

}// namespace

TEST_CASE("Shadows disappear when shadowMap.enabled is turned off") {

    auto s = makeShadowScene();

    GLRenderer renderer(glCanvas());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::PFC;

    const auto withShadows = renderOnce(renderer, s);
    REQUIRE(withShadows.size() == DATA_SIZE);
    const int darkWithShadows = countDarkPixels(withShadows);

    // Sanity: the scene must actually cast a shadow, or the test below is vacuous.
    REQUIRE(darkWithShadows > 0);

    // Turn shadows off at runtime, exactly as the settings panel does.
    renderer.shadowMap().enabled = false;
    renderer.shadowMap().needsUpdate = true;

    const auto withoutShadows = renderOnce(renderer, s);
    REQUIRE(withoutShadows.size() == DATA_SIZE);
    const int darkWithoutShadows = countDarkPixels(withoutShadows);

    INFO("dark pixels: shadows on = " << darkWithShadows
                                      << ", shadows off = " << darkWithoutShadows);

    // The shadowed region must be gone. Before the fix, `needsProgramChange` did
    // not consider the shadow config, so the material kept its USE_SHADOWMAP
    // program and went on sampling the (no longer updated) shadow map — the
    // shadow stayed on screen, merely frozen, and this count did not drop.
    CHECK(darkWithoutShadows < darkWithShadows / 2);

    renderer.dispose();
}

TEST_CASE("Shadows come back when shadowMap.enabled is turned on again") {

    auto s = makeShadowScene();

    GLRenderer renderer(glCanvas());
    renderer.shadowMap().enabled = false;

    const int darkOff = countDarkPixels(renderOnce(renderer, s));

    renderer.shadowMap().enabled = true;
    renderer.shadowMap().needsUpdate = true;

    const int darkOn = countDarkPixels(renderOnce(renderer, s));

    INFO("dark pixels: off first = " << darkOff << ", then on = " << darkOn);
    CHECK(darkOn > darkOff);

    renderer.dispose();
}

TEST_CASE("Switching shadow type at runtime does not crash") {

    // The render targets are allocated on the first shadow render, and VSM needs
    // a second target (mapPass) plus Linear filtering. Guarding that allocation
    // on `!shadow->map` alone meant a type switch after the first frame left
    // mapPass null, and the VSM blur pass dereferenced it.
    const ShadowMap order[] = {ShadowMap::PFC, ShadowMap::VSM, ShadowMap::Basic,
                               ShadowMap::VSM, ShadowMap::PFCSoft, ShadowMap::VSM};

    auto s = makeShadowScene();

    GLRenderer renderer(glCanvas());
    renderer.shadowMap().enabled = true;

    for (const auto type : order) {
        renderer.shadowMap().type = type;
        renderer.shadowMap().needsUpdate = true;

        const auto px = renderOnce(renderer, s);
        REQUIRE(px.size() == DATA_SIZE);
        // Something was drawn — a blank frame would mean the shadow pass had
        // clobbered the scene render.
        CHECK(countNonBlack(px) > 0);
    }

    renderer.dispose();
}

TEST_CASE("VSM as the very first shadow type also works") {

    // Covers the other allocation order: VSM chosen before any shadow render, so
    // both targets are created by the VSM branch up front.
    auto s = makeShadowScene();

    GLRenderer renderer(glCanvas());
    renderer.shadowMap().enabled = true;
    renderer.shadowMap().type = ShadowMap::VSM;

    const auto px = renderOnce(renderer, s);
    REQUIRE(px.size() == DATA_SIZE);
    CHECK(countNonBlack(px) > 0);

    renderer.dispose();
}

// VSM has to produce a shadow, not a moiré.
//
// The four tests above ask only whether VSM crashes or draws something, and it
// passed all of them while covering every receiver in interference fringes: the
// moments were packed into RGBA8, and with the default shadow camera spanning
// 0.5..500 a scene a few units from the light sits near depth 0.01, so the
// variance underflowed and neighbouring texels disagreed at random.
//
// Its own canvas, larger than the 64x64 shared one and with a shadow map to
// match. Both parts matter: the fringes need pixels to alternate across before
// they are measurable at all, and a map far larger than the view aliases for an
// unrelated reason (bilinear undersampling of the moments, which the variance
// test amplifies) that would mask the signal here. Roughly one shadow texel per
// pixel is also what a real frame has.
//
// Measured as a second difference along each row: zero for any smooth ramp — a
// soft shadow edge, a lit gradient — and large only where the image alternates
// pixel to pixel, which a plain gradient metric cannot tell from a soft
// penumbra. PCF is the reference for what this scene costs without fringes.
TEST_CASE("VSM renders a clean shadow, not fringes") {

    const auto alternation = [](const std::vector<unsigned char>& px, int w, int h) {
        long long energy = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 1; x < w - 1; x++) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3;
                const int prev = px[i - 3] + px[i - 2] + px[i - 1];
                const int cur = px[i] + px[i + 1] + px[i + 2];
                const int next = px[i + 3] + px[i + 4] + px[i + 5];
                energy += std::abs(2 * cur - prev - next);
            }
        }
        return energy;
    };

    constexpr int size = 256;

    Canvas canvas(Canvas::Parameters().size(size, size).headless(true));
    GLRenderer renderer(canvas);
    renderer.shadowMap().enabled = true;
    renderer.setClearColor(Color(0, 0, 0));

    auto s = makeShadowScene();
    for (auto& child : s.scene->children) {
        if (auto* d = child->as<DirectionalLight>()) d->shadow->mapSize.set(size, size);
    }

    const auto render = [&](ShadowMap type) {
        renderer.shadowMap().type = type;
        renderer.shadowMap().needsUpdate = true;
        renderer.render(*s.scene, *s.camera);
        return renderer.readRGBPixels();
    };

    const long long pcf = alternation(render(ShadowMap::PFC), size, size);

    const auto vsmPixels = render(ShadowMap::VSM);
    const long long vsm = alternation(vsmPixels, size, size);

    INFO("per-pixel alternation: PCF " << pcf << ", VSM " << vsm);

    // A soft shadow legitimately costs a little more than a hard one. Fringing
    // cost two orders of magnitude: with the moments packed into RGBA8 this
    // scene measures 2.65M against PCF's 94k.
    CHECK(vsm < pcf * 4 + 5000);

    // And it still has to cast a shadow — a VSM returning 1.0 everywhere would
    // trivially have no fringes at all.
    CHECK(countNonBlack(vsmPixels) > 0);

    int dark = 0;
    for (size_t i = 0; i < vsmPixels.size(); i += 3) {
        if (vsmPixels[i] + vsmPixels[i + 1] + vsmPixels[i + 2] < 90) dark++;
    }
    INFO("shadowed pixels: " << dark);
    CHECK(dark > 0);

    renderer.dispose();
}
