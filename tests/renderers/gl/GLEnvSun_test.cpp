// The one-sun policy on the GL raster path (Renderer::EnvSunPolicy).
//
// An equirect HDR sky bakes its sun into every GGX-prefiltered roughness strip.
// A scene that also carries a DirectionalLight (the raster convention: raster
// cannot shadow from an env map) is then lit by the sun TWICE, which is what
// made GL read brighter and warmer than the Vulkan deferred backend. The policy
// prefilters strips 1+ from a sun-clamped copy and re-injects the removed energy
// as one analytic directional light, only when the scene has no sun of its own.
//
// Synthetic env: uniform sky Le = 0.15 in every direction plus a 4°-radius disc
// at Le = 48 (under the prefilter's per-sample firefly clamp of 50, so the
// policy-Off path carries the disc's full energy and the two paths are directly
// comparable) at 45° elevation toward +Z, i.e. facing the camera.
//
// outputColorSpace=NoColorSpace + toneMapping=None → readback bytes are raw
// linear, so 1.0 ↔ 255.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/Texture.hpp"

#include <iostream>

namespace {

    constexpr int ENV_W = 512, ENV_H = 256;
    constexpr float SKY_LE = 0.15f;
    constexpr float SUN_LE = 48.f;
    constexpr float SUN_RADIUS_DEG = 4.f;

    // Unit direction TOWARD the disc centre: 45° elevation, azimuth +Z.
    Vector3 sunDirection() {
        return Vector3(0.f, std::sin(math::PI / 4), std::cos(math::PI / 4)).normalize();
    }

    std::shared_ptr<Texture> makeSunEnv() {
        const Vector3 sunDir = sunDirection();
        const float cosSun = std::cos(SUN_RADIUS_DEG * math::DEG2RAD);
        std::vector<float> data(static_cast<size_t>(ENV_W) * ENV_H * 4, 0.f);
        for (int y = 0; y < ENV_H; ++y) {
            const float elev = ((y + 0.5f) / ENV_H - 0.5f) * math::PI;
            for (int x = 0; x < ENV_W; ++x) {
                const float az = ((x + 0.5f) / ENV_W - 0.5f) * 2.f * math::PI;
                const Vector3 dir(std::cos(elev) * std::cos(az), std::sin(elev), std::cos(elev) * std::sin(az));
                const float L = dir.dot(sunDir) > cosSun ? SUN_LE : SKY_LE;
                const size_t i = (static_cast<size_t>(y) * ENV_W + x) * 4;
                data[i + 0] = data[i + 1] = data[i + 2] = L;
                data[i + 3] = 1.f;
            }
        }
        Image img{std::move(data), ENV_W, ENV_H, 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    // Diffuse-white sphere under the env alone; the centre pixel sits on the
    // +Z-facing cap, which the disc lights at cos 45°.
    std::shared_ptr<Scene> makeScene(const std::shared_ptr<Texture>& env) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = env;
        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 32), mat));
        return scene;
    }

    std::shared_ptr<PerspectiveCamera> makeCamera() {
        auto camera = PerspectiveCamera::create(45, 1.0f, 0.1f, 100.f);
        camera->position.set(0, 0, 2.4f);
        camera->lookAt(Vector3{0, 0, 0});
        return camera;
    }

    struct Shot {
        double centre = 0;
        bool found = false;
        Vector3 dir{};
    };

    Shot shoot(Scene& scene, Camera& camera, Renderer::EnvSunPolicy policy) {
        GLRenderer renderer(gltest::glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.toneMappingExposure = 1.f;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.setEnvSunPolicy(policy);
        renderer.render(scene, camera);

        auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == gltest::DATA_SIZE);

        Shot s;
        s.centre = gltest::centerPixel(px, gltest::RT_WIDTH, gltest::RT_HEIGHT).r;
        s.found = renderer.envSunFound();
        s.dir = renderer.envSunDirection();
        return s;
    }

}// namespace

// Energy conservation: what the clamp takes out of the glossy/rough strips is
// exactly what the injected analytic light puts back, so a scene with no sun of
// its own must look the SAME with the policy on and off. The residual is the
// GGX prefilter's finite-sample error integrating the raw disc in the Off path
// (the Auto path replaces that Monte-Carlo integral with a closed-form light) —
// small next to the ~2x the double sun used to add.
TEST_CASE("EnvSun: GL policy Off and Auto agree when the scene has no sun", "[envsun]") {
    auto env = makeSunEnv();
    auto scene = makeScene(env);
    auto camera = makeCamera();

    const Shot off = shoot(*scene, *camera, Renderer::EnvSunPolicy::Off);
    const Shot autoP = shoot(*scene, *camera, Renderer::EnvSunPolicy::Auto);

    std::cout << "[envsun] centre byte: off=" << off.centre << " auto=" << autoP.centre
              << "  envSunFound=" << autoP.found
              << " dir=(" << autoP.dir.x << ", " << autoP.dir.y << ", " << autoP.dir.z << ")\n";

    INFO("off " << off.centre << " vs auto " << autoP.centre);
    CHECK(std::abs(off.centre - autoP.centre) < 6.0);

    // The detector must aim the injected light at the disc it replaced: a light
    // pointing anywhere else would move the highlight off the sky's own sun.
    const Vector3 expected = sunDirection();
    const float cosErr = autoP.dir.dot(expected);
    INFO("measured dir (" << autoP.dir.x << ", " << autoP.dir.y << ", " << autoP.dir.z << ")");
    CHECK(autoP.found);
    CHECK(cosErr > std::cos(2.f * math::DEG2RAD));
}

// The one-sun rule itself: an explicit DirectionalLight claims the sun role, so
// the env goes sky-only and the disc's energy is NOT added a second time. The
// scene light here points away from the camera (intensity 0.001), so the drop
// measured on the camera-facing cap is the disc's contribution alone.
TEST_CASE("EnvSun: GL Auto stands down for a scene DirectionalLight", "[envsun]") {
    auto env = makeSunEnv();
    auto camera = makeCamera();

    auto lit = makeScene(env);
    const Shot noSceneSun = shoot(*lit, *camera, Renderer::EnvSunPolicy::Auto);

    auto withSun = makeScene(env);
    auto sceneSun = DirectionalLight::create(Color(1, 1, 1), 0.001f);
    sceneSun->position.set(0, 0, -1);// lights the far side only
    sceneSun->castShadow = false;
    withSun->add(sceneSun);
    const Shot withSceneSun = shoot(*withSun, *camera, Renderer::EnvSunPolicy::Auto);

    std::cout << "[envsun] centre byte: env-sun injected=" << noSceneSun.centre
              << " scene-sun claims it=" << withSceneSun.centre << "\n";

    // Sky-only diffuse is Le = 0.15 (~38 bytes); with the injected sun it is
    // roughly twice that. 20 bytes is a wide, unambiguous margin.
    INFO("injected " << noSceneSun.centre << " vs scene-sun " << withSceneSun.centre);
    CHECK(withSceneSun.centre < noSceneSun.centre - 20.0);
}
