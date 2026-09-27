// Where the GL renderer tone-maps (three.js r154+, #26371): only what it draws
// to the screen. A render target receives scene-linear HDR, and whatever shows
// it tone-maps once: OutputPass at the end of an EffectComposer chain, the
// Reflector shader for its mirror image.
//
// Measured in pixels, because each claim is about light that a plausible
// implementation clips or tone-maps twice while still producing a reasonable
// looking picture.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Reflector.hpp"
#include "threepp/postprocessing/postprocessing.hpp"

#include <cmath>

namespace {

    AvgColor pixelAt(const std::vector<unsigned char>& px, int x, int y) {

        const auto i = static_cast<size_t>(y * RT_WIDTH + x) * 3;
        return {static_cast<double>(px[i]), static_cast<double>(px[i + 1]), static_cast<double>(px[i + 2])};
    }

    double maxChannelDiff(const AvgColor& a, const AvgColor& b) {

        return std::max({std::abs(a.r - b.r), std::abs(a.g - b.g), std::abs(a.b - b.b)});
    }

    // Mean brightness of the pixels between `r0` and `r1` pixels from the
    // frame centre: the halo around a centred emitter, clear of the emitter.
    double ringBrightness(const std::vector<unsigned char>& px, double r0, double r1) {

        double sum = 0;
        int n = 0;
        for (int y = 0; y < RT_HEIGHT; ++y) {
            for (int x = 0; x < RT_WIDTH; ++x) {
                const double r = std::hypot(x + 0.5 - RT_WIDTH / 2.0, y + 0.5 - RT_HEIGHT / 2.0);
                if (r < r0 || r > r1) continue;
                const auto i = static_cast<size_t>(y * RT_WIDTH + x) * 3;
                sum += (px[i] + px[i + 1] + px[i + 2]) / 3.0;
                ++n;
            }
        }
        return n ? sum / n : 0;
    }

}// namespace


// A RenderPass-only chain must reproduce the direct render under a tone curve,
// including over-range light. The composer's intermediates now hold HDR, so
// this holds only if the implicit OutputPass applies the same curve, exposure
// and encode that the renderer applies inline on screen.
TEST_CASE("EffectComposer: RenderPass + implicit OutputPass equals a direct render under ACES", "[postprocessing][tonemapping]") {

    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);

    // Left half: unlit over-range colour. Right half: a lit surface.
    auto hot = MeshBasicMaterial::create();
    hot->color = Color(4.f, 1.5f, 0.3f);
    auto left = Mesh::create(PlaneGeometry::create(1, 2), hot);
    left->position.x = -0.5f;
    scene->add(left);

    auto lit = MeshStandardMaterial::create();
    lit->color = Color(0.6f, 0.6f, 0.6f);
    lit->roughness = 1.f;
    auto right = Mesh::create(PlaneGeometry::create(1, 2), lit);
    right->position.x = 0.5f;
    scene->add(right);
    auto sun = DirectionalLight::create(0xffffff, 4.f);
    sun->position.set(0, 0, 1);
    scene->add(sun);

    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10.f);
    camera->position.set(0, 0, 2);

    GLRenderer renderer(glCanvas());
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.3f;
    renderer.setClearColor(Color(0, 0, 0));
    renderer.render(*scene, *camera);
    const auto direct = renderer.readRGBPixels();

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.render();
    const auto composed = renderer.readRGBPixels();

    // One pixel in each half, clear of the seam.
    const auto dHot = pixelAt(direct, 16, 32), cHot = pixelAt(composed, 16, 32);
    const auto dLit = pixelAt(direct, 48, 32), cLit = pixelAt(composed, 48, 32);
    INFO("over-range: direct " << dHot.r << "," << dHot.g << "," << dHot.b << "  composed " << cHot.r << "," << cHot.g << "," << cHot.b);
    INFO("lit: direct " << dLit.r << "," << dLit.g << "," << dLit.b << "  composed " << cLit.r << "," << cLit.g << "," << cLit.b);
    CHECK(std::max(maxChannelDiff(dHot, cHot), maxChannelDiff(dLit, cLit)) <= 2.0);
}

// Bloom must see the scene's light, not the tone curve's output. Under inline
// tone mapping an emitter at 1 and one at 10 both reached the bloom pass near
// the curve's shoulder, so their halos were nearly the same.
TEST_CASE("UnrealBloomPass: a ten times brighter emitter blooms wider and brighter", "[postprocessing][tonemapping]") {

    auto haloFor = [](float intensity) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto material = MeshStandardMaterial::create();
        material->color = Color(0, 0, 0);
        material->emissive = Color(1, 1, 1);
        material->emissiveIntensity = intensity;
        scene->add(Mesh::create(SphereGeometry::create(0.15f, 32, 16), material));

        auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10.f);
        camera->position.set(0, 0, 2);

        GLRenderer renderer(glCanvas());
        renderer.toneMapping = ToneMapping::ACESFilmic;
        renderer.setClearColor(Color(0, 0, 0));

        EffectComposer composer(renderer);
        composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
        composer.addPass(std::make_shared<UnrealBloomPass>(Vector2(RT_WIDTH, RT_HEIGHT), 1.f, 0.4f, 0.35f));
        composer.render();

        // The sphere covers ~5 px of radius; the ring is well clear of it.
        return ringBrightness(renderer.readRGBPixels(), 10, 20);
    };

    const double dim = haloFor(1.f);
    const double bright = haloFor(10.f);

    INFO("halo at intensity 1: " << dim << "  at intensity 10: " << bright);
    CHECK(bright > 1.5 * dim);
}

// A mirror with a neutral overlay colour must show a surface as the eye sees it
// directly. The mirror texture holds un-tone-mapped linear light and the
// Reflector shader tone-maps and encodes it once; before, it was tone-mapped
// and sRGB-encoded at capture and encoded again on display.
TEST_CASE("Reflector: the mirror image of a surface matches the direct view under ACES", "[objects][tonemapping]") {

    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);

    // An unlit over-range plate behind the camera, facing the mirror.
    auto plateMaterial = MeshBasicMaterial::create();
    plateMaterial->color = Color(1.8f, 0.5f, 0.12f);
    plateMaterial->side = Side::Double;
    auto plate = Mesh::create(PlaneGeometry::create(80, 80), plateMaterial);
    plate->position.z = 5;
    scene->add(plate);

    // blendOverlay with 0.5 (linear) is the identity.
    Reflector::Options options;
    options.color = Color(0.5f, 0.5f, 0.5f);
    options.textureWidth = RT_WIDTH;
    options.textureHeight = RT_HEIGHT;
    auto mirror = Reflector::create(PlaneGeometry::create(60, 60), options);
    mirror->position.z = -5;
    scene->add(mirror);

    PerspectiveCamera camera(50, 1, 0.1f, 100);

    GLRenderer renderer(glCanvas());
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.setClearColor(Color(0, 0, 0));

    // Looking down -Z: the mirror, and in it the plate.
    camera.lookAt(Vector3(0, 0, -1));
    renderer.render(*scene, camera);
    renderer.render(*scene, camera);
    const auto reflected = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    // Turned round: the plate itself.
    camera.lookAt(Vector3(0, 0, 1));
    renderer.render(*scene, camera);
    const auto direct = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    INFO("direct " << direct.r << "," << direct.g << "," << direct.b
                   << "  reflected " << reflected.r << "," << reflected.g << "," << reflected.b);
    CHECK(maxChannelDiff(direct, reflected) <= 3.0);
}
