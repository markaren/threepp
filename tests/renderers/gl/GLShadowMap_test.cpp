// Shadow-map runtime reconfiguration.
//
// Renderer::shadowMap() is mutable at runtime (the ImGui RendererSettings panel
// exposes it), and both `enabled` and `type` feed compile-time defines
// (USE_SHADOWMAP, SHADOWMAP_TYPE_*) plus the render-target layout. Changing them
// after the first frame therefore has to invalidate material programs and, for
// VSM, reallocate the shadow targets.

#include "gl_test_helpers.hpp"

#include <catch2/generators/catch_generators.hpp>

#include <cstdlib>

#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"

namespace {

    // A lit plane with a box floating above it: the box casts a shadow onto the
    // plane, so "how dark is the darkest region" distinguishes shadowed from
    // unshadowed rendering.
    struct ShadowScene {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
    };

    enum class Sun { Directional,
                     Point };

    ShadowScene makeShadowScene(Sun sun = Sun::Directional) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        if (sun == Sun::Directional) {
            auto light = DirectionalLight::create(0xffffff, 3.f);
            light->position.set(2, 6, 2);
            light->castShadow = true;
            scene->add(light);
        } else {
            // Six cube faces packed into the viewports of one map, but the same
            // single bind-and-clear as the directional case.
            auto light = PointLight::create(0xffffff, 3.f);
            light->position.set(2, 6, 2);
            light->castShadow = true;
            scene->add(light);
        }

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

// One caster's shadow is a local feature, not a global dimming.
//
// A shadow map is cleared to WHITE - depth 1, the far plane - so a texel no
// caster wrote reads as lit. GLShadowMap set that white once before the light
// loop, but GLRenderer::setRenderTarget re-encodes the background's clear
// colour on every bind, and the bind sits between that set and the clear that
// consumes it. So the map cleared to the renderer's clear colour instead, and
// alpha carries almost all of unpackRGBAToDepth: the renderer's default
// clearAlpha of 0 meant depth 0, the near plane, and every receiver the light
// reached came back shadowed. Directional, spot and point alike - one shared
// bind. Measured here with only the map's clear changing, nothing else: mean
// frame brightness 194.3 cleared white against 68.6 stomped, out of 202.9 with
// shadows off entirely.
//
// The other tests in this file all measure the region the box covers, which
// darkens either way, and count dark pixels rather than weigh them - so a
// frame that had gone almost entirely dark still read as "a shadow is
// present". Hence a whole-frame measure here, and a ratio rather than a
// threshold: whatever the scene's absolute brightness, adding one small box to
// it may not cost three quarters of the light.
TEST_CASE("A single caster does not darken the whole frame") {

    const auto sun = GENERATE(Sun::Directional, Sun::Point);

    const auto render = [&](bool shadows) {
        auto s = makeShadowScene(sun);

        GLRenderer renderer(glCanvas());
        renderer.shadowMap().enabled = shadows;
        renderer.shadowMap().type = ShadowMap::PFC;

        renderer.render(*s.scene, *s.camera);
        auto px = renderer.readRGBPixels();
        renderer.dispose();

        return px;
    };

    const auto off = render(false);
    const auto on = render(true);

    REQUIRE(off.size() == DATA_SIZE);
    REQUIRE(on.size() == DATA_SIZE);

    // Guard: there has to be a lit scene to darken, or this passes vacuously.
    REQUIRE(avgBrightness(off) > 20.0);

    INFO("mean brightness: shadows off " << avgBrightness(off)
                                         << ", shadows on " << avgBrightness(on));

    CHECK(avgBrightness(on) > avgBrightness(off) * 0.75);

    // ...and the shadow still has to be there at all.
    CHECK(countDarkPixels(on) > countDarkPixels(off));
}

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

// The same, with a shadow map far larger than the view.
//
// The test above uses a map sized to the view, and passed while the editor -
// whose template scene keeps the 2048 default against a viewport a few hundred
// pixels wide - was covered in stipple. A pixel there spans several shadow
// texels, so the moments get point-sampled out of a map whose values ramp
// across the receiver, and neighbouring pixels land either side of the surface.
//
// This is what mipmapped moments are for, and why the map stores E[z] and
// E[z^2] rather than a mean and a deviation: a mip level of the former is still
// a valid distribution over the footprint, so the receiver can ask for the
// level that matches its own. Averaging deviations instead darkened the whole
// frustum - not stipple any more, but not a shadow either.
TEST_CASE("VSM survives a shadow map much larger than the view") {

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

    // mapSize left at its default, which is the whole point.
    auto s = makeShadowScene();

    const auto render = [&](ShadowMap type) {
        renderer.shadowMap().type = type;
        renderer.shadowMap().needsUpdate = true;
        renderer.render(*s.scene, *s.camera);
        return renderer.readRGBPixels();
    };

    const long long pcf = alternation(render(ShadowMap::PFC), size, size);

    const auto vsmPixels = render(ShadowMap::VSM);
    const long long vsm = alternation(vsmPixels, size, size);

    INFO("per-pixel alternation under minification: PCF " << pcf << ", VSM " << vsm);
    CHECK(vsm < pcf * 4 + 5000);

    // Still a shadow, and still not a uniformly dark frustum: both failure
    // modes seen while fixing this would sail past a fringe check alone.
    int dark = 0;
    for (size_t i = 0; i < vsmPixels.size(); i += 3) {
        if (vsmPixels[i] + vsmPixels[i + 1] + vsmPixels[i + 2] < 90) dark++;
    }
    const int total = static_cast<int>(vsmPixels.size() / 3);
    INFO("shadowed pixels " << dark << " of " << total);
    CHECK(dark > 0);
    CHECK(dark < total / 3);

    renderer.dispose();
}

// The r186 filter: a straight shadow edge seen from straight above.
//
// A half-plane caster over a white floor, a directional light almost overhead
// and no ambient, so a lit floor pixel and a shadowed one are the only two
// values the frame can hold unless the filter draws a penumbra between them.
// Output is left linear, so a pixel value is proportional to the light that
// reached it.
namespace {

    struct EdgeRig {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
        std::shared_ptr<DirectionalLight> light;
    };

    EdgeRig makeEdgeRig() {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto light = DirectionalLight::create(0xffffff, 2.f);
        light->position.set(0.3f, 10, 0.2f);
        light->castShadow = true;
        light->shadow->mapSize.set(256, 256);
        scene->add(light);

        auto floorMat = MeshStandardMaterial::create();
        floorMat->color = Color(1, 1, 1);
        floorMat->roughness = 1.f;
        floorMat->metalness = 0.f;
        auto floor = Mesh::create(PlaneGeometry::create(20, 20), floorMat);
        floor->rotation.x = -math::PI / 2.f;
        floor->receiveShadow = true;
        scene->add(floor);

        // Covers x < 0, above the camera: the shadow edge runs down the
        // middle of the view.
        auto casterMat = MeshStandardMaterial::create();
        auto caster = Mesh::create(BoxGeometry::create(10, 0.2f, 20), casterMat);
        caster->position.set(-5, 3, 0);
        caster->castShadow = true;
        scene->add(caster);

        auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 100.f);
        camera->position.set(0, 1.5f, 0.001f);
        camera->lookAt(Vector3{0, 0, 0});

        return {scene, camera, light};
    }

    constexpr int edgeSize = 128;

    std::vector<unsigned char> renderEdge(EdgeRig& rig, ShadowMap type) {
        Canvas canvas(Canvas::Parameters().size(edgeSize, edgeSize).headless(true));
        GLRenderer renderer(canvas);
        renderer.outputColorSpace = ColorSpace::Linear;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.shadowMap().enabled = true;
        renderer.shadowMap().type = type;
        renderer.render(*rig.scene, *rig.camera);
        auto px = renderer.readRGBPixels();
        renderer.dispose();
        return px;
    }

    int redAt(const std::vector<unsigned char>& px, int x, int y) {
        return px[(static_cast<size_t>(y) * edgeSize + x) * 3];
    }

}// namespace

TEST_CASE("PCF draws a graded penumbra across a hard edge") {

    auto rig = makeEdgeRig();
    rig.light->shadow->radius = 4;

    const auto px = renderEdge(rig, ShadowMap::PFC);
    REQUIRE(px.size() == static_cast<size_t>(edgeSize * edgeSize * 3));

    // Lit and shadowed reference values, far either side of the edge.
    const int y = edgeSize / 2;
    const int shadowed = redAt(px, 8, y);
    const int lit = redAt(px, edgeSize - 8, y);
    INFO("shadowed " << shadowed << ", lit " << lit);
    REQUIRE(lit > shadowed + 60);

    // Distinct values strictly between the two, anywhere on the middle rows:
    // a hard (or merely 2x2-bilinear) edge has at most one or two.
    std::vector<bool> seen(256, false);
    int levels = 0;
    for (int row = y - 4; row <= y + 4; ++row) {
        for (int x = 0; x < edgeSize; ++x) {
            const int v = redAt(px, x, row);
            if (v > shadowed + 8 && v < lit - 8 && !seen[v]) {
                seen[v] = true;
                ++levels;
            }
        }
    }
    INFO("intermediate levels across the edge: " << levels);
    CHECK(levels > 2);
}

TEST_CASE("shadow.intensity = 0.5 halves the darkening of a fully shadowed pixel") {

    const auto darkening = [](float intensity) {
        auto rig = makeEdgeRig();
        rig.light->shadow->intensity = intensity;
        const auto px = renderEdge(rig, ShadowMap::PFC);
        const int y = edgeSize / 2;
        const double shadowed = redAt(px, 8, y);
        const double lit = redAt(px, edgeSize - 8, y);
        return std::pair{1.0 - shadowed / lit, lit};
    };

    const auto [full, litFull] = darkening(1.f);
    const auto [half, litHalf] = darkening(0.5f);

    INFO("darkening at intensity 1: " << full << ", at 0.5: " << half << " (lit " << litFull << ")");
    REQUIRE(litFull > 60);
    REQUIRE(full > 0.9);
    CHECK(std::abs(half - 0.5 * full) < 0.05);
}

// Point lights render into a native depth cube map. Looking out from the light
// along each axis, the caster on that axis must shadow the middle of the view,
// and everywhere else - out past the 45-degree lines where one cube face hands
// over to the next - must match the same frame rendered with shadows off. A
// wrongly oriented face moves or loses its shadow; a seam, acne or a bad
// face-to-face depth conversion shows up as a difference outside the shadow.
//
// The casters are invisible to the camera (no colour, no depth) but still
// cast: the shadow pass renders them with its own depth material.
TEST_CASE("point shadows land on all six cube faces with no seam between them") {

    constexpr int size = 64;
    Canvas canvas(Canvas::Parameters().size(size, size).headless(true));

    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);

    auto roomMat = MeshStandardMaterial::create();
    roomMat->color = Color(1, 1, 1);
    roomMat->roughness = 1.f;
    roomMat->metalness = 0.f;
    roomMat->side = Side::Back;
    auto room = Mesh::create(BoxGeometry::create(8, 8, 8), roomMat);
    room->receiveShadow = true;
    scene->add(room);

    auto light = PointLight::create(0xffffff, 6.f, 0, 2);
    light->castShadow = true;
    light->shadow->mapSize.set(256, 256);
    scene->add(light);

    const Vector3 axes[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

    auto casterMat = MeshStandardMaterial::create();
    casterMat->colorWrite = false;
    casterMat->depthWrite = false;
    for (const auto& axis : axes) {
        auto caster = Mesh::create(SphereGeometry::create(0.4f, 24, 16), casterMat);
        caster->position.copy(axis).multiplyScalar(1.5f);
        caster->castShadow = true;
        scene->add(caster);
    }

    // Wide enough to see across the face boundaries at 45 degrees.
    constexpr float fov = 100;
    const float tanHalf = std::tan(math::degToRad(fov / 2));

    for (int f = 0; f < 6; ++f) {

        auto camera = PerspectiveCamera::create(fov, 1.f, 0.1f, 100.f);
        const Vector3& axis = axes[f];
        camera->up.set(std::abs(axis.y) > 0.5f ? 0.f : 1.f, 0.f, std::abs(axis.y) > 0.5f ? 1.f : 0.f);
        camera->lookAt(axis);

        const auto render = [&](bool shadows) {
            GLRenderer renderer(canvas);
            renderer.setClearColor(Color(0, 0, 0));
            renderer.shadowMap().enabled = shadows;
            renderer.shadowMap().type = ShadowMap::PFC;
            renderer.render(*scene, *camera);
            auto px = renderer.readRGBPixels();
            renderer.dispose();
            return px;
        };

        const auto on = render(true);
        const auto off = render(false);
        REQUIRE(on.size() == static_cast<size_t>(size * size * 3));

        const auto at = [&](const std::vector<unsigned char>& px, int x, int y) {
            const size_t i = (static_cast<size_t>(y) * size + x) * 3;
            return px[i] + px[i + 1] + px[i + 2];
        };

        // The caster on this axis shadows the middle of the view.
        const int centreOn = at(on, size / 2, size / 2);
        const int centreOff = at(off, size / 2, size / 2);

        // Outside a cone around the middle, shadows on and off agree.
        int worst = 0;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float u = (2.f * (static_cast<float>(x) + 0.5f) / size - 1.f) * tanHalf;
                const float v = (2.f * (static_cast<float>(y) + 0.5f) / size - 1.f) * tanHalf;
                const float angle = std::atan(std::sqrt(u * u + v * v));
                if (angle < math::degToRad(28)) continue;
                worst = std::max(worst, std::abs(at(on, x, y) - at(off, x, y)));
            }
        }

        INFO("face " << f << ": centre " << centreOn << " shadowed vs " << centreOff << " unshadowed, worst difference outside the shadow " << worst);
        CHECK(centreOn < centreOff / 4);
        CHECK(worst <= 6);
    }
}
