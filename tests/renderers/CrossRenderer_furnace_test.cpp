// White furnace parity tests — Phase 0 of the color/tone/lighting unification.
//
// Setup: a diffuse-white sphere (ρ=1, roughness=1, metalness=0) lit only by a
// uniform-radiance environment (Le=1 in every direction). For a Lambertian
// surface, L_o = (ρ/π) · ∫ L_i cos θ dω = (1/π) · Le · π = Le = 1.0, so every
// pixel on the sphere must read 1.0.
//
// outputColorSpace=NoColorSpace + toneMapping=None → readback bytes are raw
// linear, so 1.0 ↔ 255. Center pixel is always inside the sphere's projection
// (camera framing keeps it covered).
//
// Why two tests, not three: the path-tracer furnace has its own runner
// (wgpu_furnace_env_test.cpp) that needs many frames to converge; bringing
// that into a unit test would slow CI. PT parity is checked by Phase 2's gate
// against this raster reference instead.

#include "CrossRenderer_helpers.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/Texture.hpp"

namespace {

    // 8x4 equirect, Le = (1,1,1,1) per texel. Same construction as the
    // wgpu_furnace_env_test runner so any IBL math difference is exposed.
    std::shared_ptr<Texture> makeConstantEnv() {
        constexpr int W = 8, H = 4;
        std::vector<float> data(W * H * 4, 1.f);
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<Scene> makeFurnaceScene() {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = makeConstantEnv();

        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        auto sphere = Mesh::create(SphereGeometry::create(1.f, 32, 32), mat);
        scene->add(sphere);
        return scene;
    }

    std::shared_ptr<PerspectiveCamera> makeFurnaceCamera() {
        auto camera = PerspectiveCamera::create(45, 1.0f, 0.1f, 100.f);
        camera->position.set(0, 0, 2.4f);
        camera->lookAt(Vector3{0, 0, 0});
        return camera;
    }

}// namespace

TEST_CASE("Furnace: GL diffuse-white sphere in unit env reads ~1.0", "[furnace]") {
    auto scene = makeFurnaceScene();
    auto camera = makeFurnaceCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.f;
    renderer.setClearColor(Color(0, 0, 0));
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);

    auto c = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    INFO("GL center RGB: " << c.r << ", " << c.g << ", " << c.b);

    // Phase 0 target is 255 ± 1 on all three renderers; today GL is the
    // reference so we check it tightly. Raster IBL has small numerical drift
    // from PMREM sampling — 5 LSB is comfortable headroom.
    CHECK(std::abs(c.r - 255.0) < 5.0);
    CHECK(std::abs(c.g - 255.0) < 5.0);
    CHECK(std::abs(c.b - 255.0) < 5.0);
}

TEST_CASE("Furnace: Wgpu diffuse-white sphere in unit env reads ~1.0", "[furnace][wgpu]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    auto scene = makeFurnaceScene();
    auto camera = makeFurnaceCamera();

    WgpuRenderer renderer(wgpuCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.f;
    renderer.setClearColor(Color(0, 0, 0));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    REQUIRE(pixels.size() == DATA_SIZE);

    auto c = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    INFO("Wgpu center RGB: " << c.r << ", " << c.g << ", " << c.b);

    // Wider tolerance until Phase 1 (WgpuLights π-scaling + WgpuTextures sRGB
    // sampling) lands. Tighten to <5.0 once the parity gate is met.
    CHECK(std::abs(c.r - 255.0) < 30.0);
    CHECK(std::abs(c.g - 255.0) < 30.0);
    CHECK(std::abs(c.b - 255.0) < 30.0);
}
