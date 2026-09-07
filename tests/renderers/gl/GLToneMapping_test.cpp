// Tone-mapping curve tests for the GL path.
//
// An unlit MeshBasicMaterial plane fills the frame, so the fragment colour
// entering <tonemapping_fragment> is exactly the material colour. With
// outputColorSpace=NoColorSpace the readback byte is the tone mapper's linear
// output × 255 — i.e. the curve is measured directly, with no sRGB encode in
// the way.
//
// Targets are closed-form evaluations of the operator, not captured
// references, so a transposed matrix or a swapped constant fails loudly.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"

namespace {

    // Full-frame unlit plate. Ortho camera at z=2, plane 4×4 → every pixel of
    // the 64² target is covered.
    double toneMappedByte(float linearValue, ToneMapping mode, float exposure = 1.f) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto mat = MeshBasicMaterial::create();
        mat->color = Color(linearValue, linearValue, linearValue);
        scene->add(Mesh::create(PlaneGeometry::create(4, 4), mat));

        auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10.f);
        camera->position.set(0, 0, 2);
        camera->lookAt(Vector3{0, 0, 0});

        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = mode;
        renderer.toneMappingExposure = exposure;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*scene, *camera);

        auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == DATA_SIZE);
        return centerPixel(px, RT_WIDTH, RT_HEIGHT).r;
    }

}// namespace

// AgX on white. Both AgX matrices and both Rec2020 matrices have unit row
// sums, so an achromatic input stays achromatic all the way through and the
// whole chain collapses to a scalar:
//   log2(1) = 0 → t = 12.47393/16.5 = 0.7560
//   sigmoid(0.7560) = 0.78678  (2.2-gamma domain)
//   0.78678^2.2 = 0.59025      → byte 150
// A transposed inset/outset matrix breaks the unit row sum and lands
// elsewhere; a swapped AgxMin/MaxEv moves t and therefore the sigmoid.
TEST_CASE("ToneMapping: GL AgX maps linear 1.0 to byte 150", "[tonemapping]") {
    const double c = toneMappedByte(1.f, ToneMapping::AgX);
    INFO("AgX(1.0) analytic 150, GL: " << c);
    CHECK(std::abs(c - 150.0) < 2.0);
}

// Middle grey is the case that actually pins the log2 encode: 0.18 sits at
// t = 10/16.5 = 0.60606 on the AgX Ev range, sigmoid → 0.49674, and
// 0.49674^2.2 = 0.21451 → byte 55. AgX is defined by putting middle grey at
// ≈0.497 in the 2.2-gamma display domain; that is what this measures.
TEST_CASE("ToneMapping: GL AgX puts middle grey at ~0.497 display", "[tonemapping]") {
    const double c = toneMappedByte(0.18f, ToneMapping::AgX);
    INFO("AgX(0.18) analytic 55, GL: " << c);
    CHECK(std::abs(c - 55.0) < 2.0);
}

// AgX must not be silently falling through to Linear (the pre-C1 behaviour:
// GLProgram hit `default:`, printed a warning and emitted LinearToneMapping).
// Linear would read 255 and 46 for these two inputs.
TEST_CASE("ToneMapping: GL AgX is not the Linear fallback", "[tonemapping]") {
    CHECK(toneMappedByte(1.f, ToneMapping::Linear) > 250.0);
    CHECK(toneMappedByte(1.f, ToneMapping::AgX) < 200.0);
}
