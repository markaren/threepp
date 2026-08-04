// Direct-specular BRDF tests for the GL physical material.
//
// SHARED FIXTURE — chosen so the whole lobe collapses to a closed form:
//   * orthographic camera at (0,0,2) looking at the origin. Under ortho
//     <lights_fragment_begin> sets geometry.viewDir = vec3(0,0,1) for EVERY
//     pixel, with no per-pixel view vector to reason about.
//   * a 4x4 plane facing +Z fills the 64x64 target (the ortho frustum is
//     [-1,1]^2), so every pixel is the same shading point.
//   * one directional light at (0,0,5) aimed at the origin -> L = (0,0,1).
//   * material colour BLACK, so diffuseColor = 0 and the diffuse lobe drops
//     out entirely; metalness 0 keeps F0 off the albedo.
//
// With N = V = L = H every dot product is 1, so:
//   D_GGX( 0.25, 1 ) * G_GGX_SmithCorrelated( 0.25, 1, 1 ) = (16/PI) * 0.25 = 4/PI
//   F_Schlick( f0, f90, 1 ) = f0   exactly, for ANY f90
// and useLegacyLights defaults false, so there is no extra factor of PI.
//
//   ==> every pixel = intensity * F0 * 4/PI
//
// outputColorSpace=NoColorSpace + toneMapping=None makes the readback byte
// that value * 255 directly.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshPhysicalMaterial.hpp"

namespace {

    constexpr double FOUR_OVER_PI = 4.0 / 3.14159265358979323846;

    struct Fixture {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<OrthographicCamera> camera;
        std::shared_ptr<MeshPhysicalMaterial> material;
    };

    Fixture makeFixture(float intensity) {
        Fixture f;
        f.scene = Scene::create();
        f.scene->background = Color(0, 0, 0);

        f.material = MeshPhysicalMaterial::create();
        f.material->color = Color(0, 0, 0);// kills the diffuse lobe
        f.material->roughness = 0.5f;      // alpha = 0.25
        f.material->metalness = 0.f;
        f.scene->add(Mesh::create(PlaneGeometry::create(4, 4), f.material));

        auto light = DirectionalLight::create(Color(0xffffff), intensity);
        light->position.set(0, 0, 5);
        f.scene->add(light);

        f.camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10.f);
        f.camera->position.set(0, 0, 2);
        f.camera->lookAt(Vector3{0, 0, 0});
        return f;
    }

    AvgColor renderFixture(const Fixture& f) {
        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.toneMappingExposure = 1.f;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*f.scene, *f.camera);
        auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == DATA_SIZE);
        return centerPixel(px, RT_WIDTH, RT_HEIGHT);
    }

}// namespace

// The pow5 switch, isolated.
//
// reflectivity = 0.05 gives F0 = MAXIMUM_SPECULAR_COEFFICIENT * 0.05^2
// = 0.16 * 0.0025 = 4.0e-4, a deliberately tiny F0. At intensity 1000:
//   exact pow5:  F = f0                 -> 1000 * 4.0e-4  * 4/PI = 0.5093 -> byte 130
//   Epic's exp2: F = f0 + (1-f0)*2^-12.54 = 5.68e-4       -> 0.7233 -> byte 184
// The exp2 fit does not reach 0 at dotVH = 1; that residual is normally
// sub-LSB, and this fixture is built to magnify it into 54 LSB. Fails hard on
// the old Fresnel, so it also guards against an exp2 site being reintroduced.
TEST_CASE("BRDF: GL direct specular uses the exact pow5 Schlick", "[brdf]") {
    auto f = makeFixture(1000.f);
    f.material->reflectivity = 0.05f;

    const auto c = renderFixture(f);
    const double expected = 1000.0 * 4.0e-4 * FOUR_OVER_PI * 255.0;// 129.9
    INFO("F0=4e-4 pow5 analytic " << expected << " (exp2 fit would read ~184), GL: " << c.r);
    CHECK(std::abs(c.r - expected) < 2.0);
}

// Sanity anchor at the default F0 = 0.04 (reflectivity 0.5): the fixture's
// closed form has to hold at an ordinary F0 too, not just the tiny one above.
//   18 * 0.04 * 4/PI = 0.9167 -> byte 234
TEST_CASE("BRDF: GL direct specular matches the closed form at default F0", "[brdf]") {
    auto f = makeFixture(18.f);

    const auto c = renderFixture(f);
    const double expected = 18.0 * 0.04 * FOUR_OVER_PI * 255.0;// 233.7
    INFO("F0=0.04 analytic " << expected << ", GL: " << c.r);
    CHECK(std::abs(c.r - expected) < 2.0);
}
