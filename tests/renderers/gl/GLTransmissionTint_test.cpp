// What comes THROUGH a transmissive surface is filtered by its base colour.
//
// KHR_materials_transmission: "the base color filters light passing through", and
// three.js multiplies the transmitted sample by the diffuse colour. The GL
// transmission chunk lerped a near-black tint (max channel < 0.1) back to WHITE,
// on the theory that a black base colour means clear glass. glTF sample
// SunglassesKhronos has lenses at 0.009 and 0.016, alphaMode OPAQUE: they were
// clear from both sides.
//
// The assets the lerp was written for pair a black base colour with alphaMode
// BLEND and a low alpha (smoked car windows). What they ask for is coverage:
//   a * [ (1-t) * diffuse + t * tint * behind ] + (1-a) * behind
// which the chunk now folds into its transmission and tint (the pass writes
// alpha = 1), as the Vulkan host does.
//
// Setup: an unlit white wall, a transmissive quad in front of it, no environment
// and ior 1.0, so the quad has no reflection of its own at normal incidence and
// the centre pixel is simply tint * wall. Raw linear readback: wall = 255.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"

using namespace gltest;

namespace {

    double throughGlass(const std::shared_ptr<MeshPhysicalMaterial>& glass) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto wallMat = MeshBasicMaterial::create();
        wallMat->color = Color(1, 1, 1);
        auto wall = Mesh::create(PlaneGeometry::create(20.f, 20.f), wallMat);
        wall->position.z = -2.f;
        scene->add(wall);

        scene->add(Mesh::create(PlaneGeometry::create(1.f, 1.f), glass));

        auto camera = PerspectiveCamera::create(45, 1.f, 0.1f, 100.f);
        camera->position.set(0, 0, 1.f);// the quad overfills the frame
        camera->lookAt(Vector3{0, 0, 0});

        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*scene, *camera);
        return centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT).r;
    }

    std::shared_ptr<MeshPhysicalMaterial> glassOf(float grey) {
        auto m = MeshPhysicalMaterial::create();
        m->color = Color(grey, grey, grey);
        m->metalness = 0.f;
        m->roughness = 0.f;
        m->transmission = 1.f;
        m->setIor(1.0f);
        return m;
    }

}// namespace

TEST_CASE("GL transmission: a dark base colour is dark glass", "[transmission]") {

    // Control: a mid grey was tinted before the fix too (the lerp ended at 0.1).
    const double mid = throughGlass(glassOf(0.5f));
    INFO("grey 0.5 -> " << mid << " (expected 127.5)");
    CHECK(std::abs(mid - 127.5) < 6.0);

    // The sunglasses' lens. Was 0.97 of the wall: the tint lerped to white.
    const double dark = throughGlass(glassOf(0.01f));
    INFO("grey 0.01 -> " << dark << " (expected 2.6)");
    CHECK(dark < 12.0);
}

TEST_CASE("GL transmission: alpha on a blended glass is coverage", "[transmission]") {

    // Black glass at alpha 0.25: a quarter of the pixel is black glass, the
    // rest is the wall. 0.75 * 255 = 191.
    auto smoked = glassOf(0.f);
    smoked->transparent = true;
    smoked->opacity = 0.25f;
    const double v = throughGlass(smoked);
    INFO("black glass, alpha 0.25 -> " << v << " (expected 191)");
    CHECK(std::abs(v - 191.0) < 8.0);

    // The same material NOT blended: glTF ignores alpha in OPAQUE mode.
    auto solid = glassOf(0.f);
    solid->opacity = 0.25f;
    const double o = throughGlass(solid);
    INFO("black glass, alpha 0.25, not blended -> " << o << " (expected 0)");
    CHECK(o < 8.0);
}
