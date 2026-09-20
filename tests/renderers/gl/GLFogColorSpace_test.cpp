// Which colour space the fog uniform is encoded into.
//
// <fog_fragment> mixes fogColor in AFTER <encodings_fragment>, i.e. in output
// space, so the uniform has to be pre-encoded — that is what makes a fully fogged
// surface land on the authored value and meet a Color background with no horizon
// line. But WHICH encode depends on where the pass is drawing: the shader's
// linearToOutputTexel is compiled from the bound target's colour space, and a
// RenderTarget defaults to NoColorSpace, for which that transform is the
// identity.
//
// refreshFogUniforms encoded to sRGB unconditionally, so every offscreen pass —
// an EffectComposer's ping-pong targets, the transmission backdrop, a mirror —
// got an sRGB value written into a linear buffer: roughly three times too bright,
// and out of step with the background clear, which GLBackground already chooses
// from the bound target (GLBackground::setClear).
//
// A black plane far beyond the fog's far distance reads as pure fog, so these two
// renders of the SAME scene must differ by exactly one sRGB encode.

#include "gl_test_helpers.hpp"

#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Fog.hpp"

namespace {

    constexpr int kFogHex = 0x6688aa;

    struct Fixture {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
    };

    Fixture makeFixture() {
        Fixture f;
        f.scene = Scene::create();
        f.scene->fog = Fog(Color(kFogHex), 1.f, 20.f);

        auto mat = MeshBasicMaterial::create();
        mat->color = Color(0x000000);
        f.scene->add(Mesh::create(PlaneGeometry::create(400.f, 400.f), mat));

        f.camera = PerspectiveCamera::create(50, 1.f, 0.1f, 1000.f);
        f.camera->position.set(0, 0, 60.f);
        f.camera->lookAt(Vector3{0, 0, 0});
        return f;
    }

    double srgbToLinear(double c) {
        c /= 255.0;
        const double lin = c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
        return lin * 255.0;
    }

}// namespace

TEST_CASE("fog is encoded for the target it is drawn into") {

    auto f = makeFixture();

    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(kFogHex));

    // 1. Straight to the screen, whose output space is sRGB. Fully fogged pixels
    //    must read the authored bytes.
    renderer.render(*f.scene, *f.camera);
    const auto onScreen = averageColor(renderer.readRGBPixels());

    INFO("on screen: " << onScreen.r << ", " << onScreen.g << ", " << onScreen.b);
    CHECK(std::abs(onScreen.r - 102.0) < 4.0);
    CHECK(std::abs(onScreen.g - 136.0) < 4.0);
    CHECK(std::abs(onScreen.b - 170.0) < 4.0);

    // 2. The same scene into a RenderTarget left at its default colour space,
    //    where the shader's output transform is the identity. The buffer is
    //    linear, so fully fogged pixels must read the LINEAR value.
    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*f.scene, *f.camera);
    const auto offScreen = averageColor(renderer.readRGBPixels());
    renderer.setRenderTarget(nullptr);

    const double wantR = srgbToLinear(102.0);// ~34
    const double wantG = srgbToLinear(136.0);// ~63
    const double wantB = srgbToLinear(170.0);// ~102

    INFO("into a linear target: " << offScreen.r << ", " << offScreen.g << ", " << offScreen.b
                                  << "  (linear fog colour is " << wantR << ", " << wantG << ", " << wantB
                                  << "; the sRGB bytes would be 102, 136, 170)");
    CHECK(std::abs(offScreen.r - wantR) < 5.0);
    CHECK(std::abs(offScreen.g - wantG) < 5.0);
    CHECK(std::abs(offScreen.b - wantB) < 5.0);
}
