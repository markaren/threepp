// Shadow-map runtime reconfiguration.
//
// Renderer::shadowMap() is mutable at runtime (the ImGui RendererSettings panel
// exposes it), and both `enabled` and `type` feed compile-time defines
// (USE_SHADOWMAP, SHADOWMAP_TYPE_*) plus the render-target layout. Changing them
// after the first frame therefore has to invalidate material programs and, for
// VSM, reallocate the shadow targets.

#include "gl_test_helpers.hpp"

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
