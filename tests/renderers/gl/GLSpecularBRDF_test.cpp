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

    // 8x4 equirect, Le = 1 per texel. Used by the F90 case, which needs an
    // environment: F90 only shows up in the split-sum's brdf.y term, and the
    // analytic-light fixture above evaluates Fresnel at dotVH = 1 where F90
    // has zero weight by construction.
    std::shared_ptr<Texture> makeConstantEnv(float Le) {
        constexpr int W = 8, H = 4;
        std::vector<float> data(W * H * 4, Le);
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

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

    AvgColor renderWith(GLRenderer& renderer, const Fixture& f) {
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.toneMappingExposure = 1.f;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*f.scene, *f.camera);
        auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == DATA_SIZE);
        return centerPixel(px, RT_WIDTH, RT_HEIGHT);
    }

    AvgColor renderFixture(const Fixture& f) {
        GLRenderer renderer(glCanvas());
        return renderWith(renderer, f);
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

// --- KHR_materials_specular -------------------------------------------------

// specularIntensity scales F0 linearly. At dotVH = 1 the direct lobe is exactly
// F0, so halving the intensity has to halve the byte: 234 -> 117, no rounding
// slack to hide behind.
TEST_CASE("KHR specular: GL specularIntensity scales F0 linearly", "[brdf][specular]") {
    auto f = makeFixture(18.f);
    f.material->specularIntensity = 0.5f;

    const auto c = renderFixture(f);
    const double expected = 18.0 * 0.04 * 0.5 * FOUR_OVER_PI * 255.0;// 116.8
    INFO("intensity 0.5 analytic " << expected << ", GL: " << c.r);
    CHECK(std::abs(c.r - expected) < 2.0);
}

// specularColor tints F0 per channel. A (1, 0.5, 0.25) tint must come back as
// an exact 4:2:1 ratio — this is what catches the tint being applied to the
// wrong term (e.g. multiplied into F90, or after the metalness mix).
TEST_CASE("KHR specular: GL specularColor tints F0 per channel", "[brdf][specular]") {
    auto f = makeFixture(18.f);
    f.material->specularColor = Color(1.f, 0.5f, 0.25f);

    const auto c = renderFixture(f);
    const double base = 18.0 * 0.04 * FOUR_OVER_PI * 255.0;
    INFO("tint (1,.5,.25) analytic " << base << "/" << base / 2 << "/" << base / 4
                                     << ", GL: " << c.r << ", " << c.g << ", " << c.b);
    CHECK(std::abs(c.r - base) < 2.0);
    CHECK(std::abs(c.g - base / 2.0) < 2.0);
    CHECK(std::abs(c.b - base / 4.0) < 2.0);
}

// THE F90 CASE. F90 lives only in the split-sum environment BRDF's brdf.y
// term, which BRDF_Specular_GGX_Environment used to add unweighted. With
// specularIntensity = 0 the surface has no specular response at all, so an
// env-lit black dielectric must read ~0. If brdf.y is still unweighted it
// reads ~9 instead — the whole grazing lobe survives an intensity of zero.
//
// Le = 20 stays under the PMREM firefly clamp; the material is black and
// metalness 0, so the diffuse lobe contributes nothing and every photon in the
// frame came through the specular path.
TEST_CASE("KHR specular: GL specularIntensity 0 kills the env specular lobe", "[brdf][specular]") {
    auto f = makeFixture(0.f);// no analytic light: env only
    f.scene->environment = makeConstantEnv(20.f);

    const double lit = renderFixture(f).r;

    f.material->specularIntensity = 0.f;
    f.material->needsUpdate();
    const double dark = renderFixture(f).r;

    INFO("env specular lit: " << lit << ", intensity 0: " << dark);
    CHECK(lit > 5.0);  // the lobe is actually doing something to begin with
    CHECK(dark < 1.51);// ...and intensity 0 switches all of it off
}

// --- KHR_materials_iridescence ---------------------------------------------
//
// evalIridescence is hard to pin with a hand-computed number at a general
// thickness, but there is one regime where it is trivial: as thickness grows,
// the exp( -phase^2 * vr ) factor inside iridSensitivity annihilates both
// higher-order Fourier terms and the whole function collapses to its
// achromatic zeroth term C0 = R12 + Rs.
//
// At 4000 nm that factor is ~9e-9, so with IOR 1.3 over the default F0 = 0.04,
// at normal incidence (cosTheta1 = 1, which the ortho fixture guarantees):
//   R12  = ((1.3-1)/(1.3+1))^2                       = 0.0170132
//   base IOR from F0=0.04 = (1+0.2)/(1-0.2)          = 1.5
//   R23  = ((1.5-1.3)/(1.5+1.3))^2                   = 0.0051020
//   T121 = 1 - R12,  Rs = T121^2*R23/(1-R12*R23)     = 0.0049293
//   C0   = R12 + Rs                                  = 0.0219425
// against a base F0 of 0.04. Both fall straight out of the fixture's
// intensity * F * 4/PI, so the mix weight can be read directly off the byte.
namespace {

    constexpr double IRID_C0_4000NM = 0.0219425;

    double byteFor(double F) {
        return 18.0 * F * FOUR_OVER_PI * 255.0;
    }

}// namespace

TEST_CASE("Iridescence: GL thin film collapses to its achromatic C0 at 4000nm", "[brdf][iridescence]") {
    auto f = makeFixture(18.f);
    f.material->iridescence = 1.f;
    f.material->iridescenceIOR = 1.3f;
    f.material->iridescenceThicknessNm = 4000.f;

    const auto c = renderFixture(f);
    const double expected = byteFor(IRID_C0_4000NM);// 128.2
    INFO("C0 analytic " << expected << ", GL: " << c.r << ", " << c.g << ", " << c.b);
    CHECK(std::abs(c.r - expected) < 2.0);
    // Achromatic: the spectral terms really are gone, not merely small.
    CHECK(std::abs(c.r - c.g) <= 1.0);
    CHECK(std::abs(c.g - c.b) <= 1.0);
}

// The mix weight itself. Halfway between base F0 (byte 234) and the thin-film
// C0 (byte 128) has to land on 181 — a weight applied to the wrong operand, or
// applied twice, misses this.
TEST_CASE("Iridescence: GL mixes the thin-film Fresnel by the layer weight", "[brdf][iridescence]") {
    auto f = makeFixture(18.f);
    f.material->iridescence = 0.5f;
    f.material->iridescenceIOR = 1.3f;
    f.material->iridescenceThicknessNm = 4000.f;

    const auto c = renderFixture(f);
    const double expected = byteFor(0.5 * 0.04 + 0.5 * IRID_C0_4000NM);// 181.0
    INFO("half-mix analytic " << expected << ", GL: " << c.r);
    CHECK(std::abs(c.r - expected) < 2.0);
}

// A thickness of 0 means "no film". The shader forces the layer weight to zero
// in that case, so the pixel must equal the plain dielectric — while STILL
// compiling the USE_IRIDESCENCE variant, since the program flag is gated on
// iridescence > 0 and this material has iridescence = 1.
//
// That makes this simultaneously a cache-key check: both draws go through one
// GLRenderer, and a hash() that ignored the iridescence flag would serve the
// first program for the second material.
TEST_CASE("Iridescence: GL zero thickness is exactly the plain dielectric", "[brdf][iridescence]") {
    GLRenderer renderer(glCanvas());

    auto plain = makeFixture(18.f);
    const double base = renderWith(renderer, plain).r;

    auto zeroThickness = makeFixture(18.f);
    zeroThickness.material->iridescence = 1.f;
    zeroThickness.material->iridescenceIOR = 1.3f;
    zeroThickness.material->iridescenceThicknessNm = 0.f;
    const double zero = renderWith(renderer, zeroThickness).r;

    auto film = makeFixture(18.f);
    film.material->iridescence = 1.f;
    film.material->iridescenceIOR = 1.3f;
    film.material->iridescenceThicknessNm = 4000.f;
    const double lit = renderWith(renderer, film).r;

    INFO("plain " << base << ", thickness 0 " << zero << ", 4000nm film " << lit);
    CHECK(base == zero);      // byte-identical through a different program variant
    CHECK(std::abs(lit - zero) > 50.0);// ...and the variant is not being ignored
}

// The achromatic anchor above cannot see a bungled iridSensitivity: the whole
// spectral block is multiplied by ~0 at 4000 nm. At 550 nm it dominates, and
// the film reads strongly blue — analytically about (23, 91, 171). Assert a
// large channel spread rather than the exact triple, so a plausible-but-not-
// bit-exact rasterizer still passes while a transposed XYZ_TO_REC709 or a
// mistyped coefficient (both of which flatten or invert the hue) does not.
TEST_CASE("Iridescence: GL 550nm film is strongly chromatic", "[brdf][iridescence]") {
    auto f = makeFixture(18.f);
    f.material->iridescence = 1.f;
    f.material->iridescenceIOR = 1.3f;
    f.material->iridescenceThicknessNm = 550.f;

    const auto c = renderFixture(f);
    const double spread = std::max({c.r, c.g, c.b}) - std::min({c.r, c.g, c.b});
    INFO("550nm film RGB: " << c.r << ", " << c.g << ", " << c.b << " (spread " << spread << ")");
    CHECK(spread > 15.0);
    // Blue-dominant, as the Fourier terms put it — catches an inverted hue.
    CHECK(c.b > c.g);
    CHECK(c.g > c.r);
}

// The IBL half of the feature, and the only thing that exercises
// Schlick_to_F0. The split-sum environment BRDF consumes an F0, not a
// reflectance, so the thin-film Fresnel is folded back through the exact
// inverse of Schlick before being mixed into the IBL F0.
//
// At normal incidence Schlick_to_F0 is the identity (x5 = 0), which makes this
// closed-form too. With the 4000 nm achromatic film over Le = 20:
//   integrateSpecularBRDF( 1, 0.5 ) = ( 0.723263, 0.001737 )
//   plain: 20 * ( 0.04     * 0.723263 + 0.001737 ) = 0.6133 -> byte 156
//   film:  20 * ( 0.021943 * 0.723263 + 0.001737 ) = 0.3521 -> byte  90
// A missing Schlick_to_F0 (feeding the raw Fresnel straight in) happens to
// agree here BY CONSTRUCTION at dotNV = 1; what this pins is that the IBL lobe
// mixes at all rather than ignoring iridescence, which it did before.
TEST_CASE("Iridescence: GL env specular uses the folded iridescence F0", "[brdf][iridescence]") {
    auto plain = makeFixture(0.f);
    plain.scene->environment = makeConstantEnv(20.f);
    const double base = renderFixture(plain).r;

    auto film = makeFixture(0.f);
    film.scene->environment = makeConstantEnv(20.f);
    film.material->iridescence = 1.f;
    film.material->iridescenceIOR = 1.3f;
    film.material->iridescenceThicknessNm = 4000.f;
    const double lit = renderFixture(film).r;

    const double brdfX = 0.723263, brdfY = 0.001737;
    const double expectedBase = 20.0 * (0.04 * brdfX + brdfY) * 255.0;
    const double expectedFilm = 20.0 * (IRID_C0_4000NM * brdfX + brdfY) * 255.0;
    INFO("env plain analytic " << expectedBase << " GL " << base
                               << " | env film analytic " << expectedFilm << " GL " << lit);
    CHECK(std::abs(base - expectedBase) < 3.0);
    CHECK(std::abs(lit - expectedFilm) < 3.0);
}
