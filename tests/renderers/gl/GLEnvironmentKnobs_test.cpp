// three.js r146-r163 scene knobs on the GL backend: environmentIntensity,
// backgroundIntensity, and a plain 2D texture background (which the GL
// background used to ignore silently).
//
// Output is read with NoColorSpace and no tone mapping, so a pixel is the
// linear value times 255 and "half" means half.

#include "gl_test_helpers.hpp"

#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"
#include "threepp/textures/Texture.hpp"

using namespace gltest;

namespace {

    std::shared_ptr<Texture> makeConstantEquirect(float value) {
        constexpr int W = 8, H = 4;
        std::vector<float> data(W * H * 4, value);
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<PerspectiveCamera> makeCamera() {
        auto camera = PerspectiveCamera::create(45, 1, 0.1f, 100);
        camera->position.set(0, 0, 3);
        camera->lookAt(0, 0, 0);
        return camera;
    }

    void linearOutput(GLRenderer& renderer) {
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
    }

}// namespace

TEST_CASE("environmentIntensity 0 turns a RoomEnvironment-lit metal sphere off", "[gl][environment]") {

    auto camera = makeCamera();

    auto render = [&](bool withEnvironment, float environmentIntensity) {
        GLRenderer renderer(glCanvas());
        linearOutput(renderer);

        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        auto mat = MeshStandardMaterial::create();
        mat->metalness = 1.f;
        mat->roughness = 0.4f;
        scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 32), mat));

        // The environment is GPU-rendered from a scene, so this also runs the
        // whole PMREMGenerator::fromScene path (and its skipped sun detector).
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        if (withEnvironment) scene->environment = pmrem.fromScene(room, 0.04f, 0.1f, 100.f, {64, {}});
        scene->environmentIntensity = environmentIntensity;

        renderer.render(*scene, *camera);
        return renderer.readRGBPixels();
    };

    const auto none = render(false, 1.f);
    const auto lit = render(true, 1.f);
    const auto off = render(true, 0.f);

    // The environment must actually light the sphere, or "off" proves nothing.
    INFO("lit avg " << avgBrightness(lit) << ", none avg " << avgBrightness(none));
    REQUIRE(avgBrightness(lit) > avgBrightness(none) + 20.0);

    int differing = 0;
    for (size_t i = 0; i < off.size(); ++i) {
        if (std::abs(static_cast<int>(off[i]) - static_cast<int>(none[i])) > 1) ++differing;
    }
    INFO("channels differing from the no-environment render: " << differing);
    REQUIRE(differing == 0);
}

TEST_CASE("backgroundIntensity 0.5 halves a texture background", "[gl][background]") {

    auto camera = makeCamera();

    auto render = [&](float backgroundIntensity) {
        GLRenderer renderer(glCanvas());
        linearOutput(renderer);
        Scene scene;
        scene.background = Background(makeConstantEquirect(0.8f));
        scene.backgroundIntensity = backgroundIntensity;
        renderer.render(scene, *camera);
        return centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);
    };

    const auto full = render(1.f);
    const auto half = render(0.5f);
    INFO("full " << full.r << ", half " << half.r);
    REQUIRE(std::abs(full.r - 0.8 * 255) < 3.0);
    REQUIRE(std::abs(half.r - 0.5 * full.r) < 2.0);
}

TEST_CASE("a plain 2D texture background is drawn", "[gl][background]") {

    GLRenderer renderer(glCanvas());
    linearOutput(renderer);

    std::vector<unsigned char> data;
    for (int i = 0; i < 4; ++i) data.insert(data.end(), {200, 50, 30, 255});
    auto tex = Texture::create(Image(std::move(data), 2, 2));
    tex->format = Format::RGBA;
    tex->colorSpace = ColorSpace::Linear;
    tex->needsUpdate();

    Scene scene;
    scene.background = Background(tex);
    auto camera = makeCamera();
    renderer.render(scene, *camera);

    const auto pixels = renderer.readRGBPixels();
    INFO("average " << averageColor(pixels).r << ", " << averageColor(pixels).g << ", " << averageColor(pixels).b);
    REQUIRE(allPixelsMatch(pixels, 200, 50, 30, 2));
}
