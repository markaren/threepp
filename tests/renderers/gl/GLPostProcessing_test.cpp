// EffectComposer behaviour on the GL path, measured in pixels.
//
// Every claim here is about something the chain is supposed to guarantee and
// that a plausible-looking implementation gets wrong: that a pass actually runs
// (and runs in order), that the composer's output is sRGB-encoded rather than
// left linear in the target, that ping-pong buffers do not alias, that a
// disabled pass is skipped, that a MaskPass confines what follows it, and that
// multisampled composer targets resolve.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/postprocessing/postprocessing.hpp"

namespace {

    // Full-frame unlit plate of a known linear colour.
    std::shared_ptr<Scene> platedScene(const Color& color) {

        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto mat = MeshBasicMaterial::create();
        mat->color = color;
        scene->add(Mesh::create(PlaneGeometry::create(4, 4), mat));

        return scene;
    }

    std::shared_ptr<OrthographicCamera> plateCamera() {

        auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10.f);
        camera->position.set(0, 0, 2);
        camera->lookAt(Vector3{0, 0, 0});

        return camera;
    }

    // Multiplies the incoming image by a constant, so a pass's presence and
    // ordering are both readable off a single pixel.
    Shader scaleShader(float scale) {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"scale", Uniform(scale)}},

                R"(
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform sampler2D tDiffuse;
                uniform float scale;
                varying vec2 vUv;
                void main() {
                    gl_FragColor = vec4( texture2D( tDiffuse, vUv ).rgb * scale, 1.0 );
                })"};
    }

}// namespace


// The chain has to actually run: a RenderPass alone must put the scene on
// screen, and it must arrive through the composer's own output draw rather than
// from whatever the renderer last drew.
TEST_CASE("EffectComposer: RenderPass alone reproduces a direct render", "[postprocessing]") {

    auto scene = platedScene(Color(0.25f, 0.5f, 0.75f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));
    renderer.render(*scene, *camera);
    const auto direct = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    INFO("direct " << direct.r << "," << direct.g << "," << direct.b
                   << "  composed " << composed.r << "," << composed.g << "," << composed.b);
    CHECK(std::abs(composed.r - direct.r) < 2.0);
    CHECK(std::abs(composed.g - direct.g) < 2.0);
    CHECK(std::abs(composed.b - direct.b) < 2.0);
}

// The trap this pins: intermediate targets are linear, and threepp compiles the
// output transform against whichever target is bound. A chain that hands the
// screen to a user pass would skip the sRGB encode and land ~0.5 instead of
// ~0.73 — the classic washed-out composer image. The composer owning its final
// draw is what keeps the two paths equal.
TEST_CASE("EffectComposer: output is sRGB-encoded like a direct render", "[postprocessing]") {

    auto scene = platedScene(Color(0.5f, 0.5f, 0.5f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.setClearColor(Color(0, 0, 0));

    renderer.render(*scene, *camera);
    const auto direct = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    // sRGB encode of linear 0.5 is 0.7354 -> byte 188. Linear would be 128, and
    // encoding twice lands at 223.
    INFO("direct " << direct.r << ", composed " << composed.r << " (expected ~188)");
    CHECK(std::abs(direct.r - 188.0) < 3.0);
    CHECK(std::abs(composed.r - 188.0) < 3.0);
}

// Two passes, and the second reads what the first wrote. If the ping-pong
// buffers aliased, or a swap were missed, the product would not be the product.
TEST_CASE("EffectComposer: passes compose in order", "[postprocessing]") {

    auto scene = platedScene(Color(0.8f, 0.8f, 0.8f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.addPass(std::make_shared<ShaderPass>(scaleShader(0.5f)));
    composer.addPass(std::make_shared<ShaderPass>(scaleShader(0.5f)));
    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    // 0.8 * 0.5 * 0.5 = 0.2 -> byte 51.
    INFO("expected ~51, got " << composed.r);
    CHECK(std::abs(composed.r - 51.0) < 3.0);
}

// A disabled pass must be skipped whole — including its buffer swap. Skipping
// the draw but not the swap is the classic way to get a black frame instead.
TEST_CASE("EffectComposer: a disabled pass is skipped, not half-run", "[postprocessing]") {

    auto scene = platedScene(Color(0.8f, 0.8f, 0.8f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    auto scalePass = std::make_shared<ShaderPass>(scaleShader(0.5f));
    scalePass->enabled = false;
    composer.addPass(scalePass);

    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    // Untouched 0.8 -> byte 204.
    INFO("expected ~204, got " << composed.r);
    CHECK(std::abs(composed.r - 204.0) < 3.0);
}

// Rendering twice must not drift. The buffers alternate roles across frames
// (nothing resets the read/write assignment), so an odd-length chain reading a
// stale target shows up as a second frame that differs from the first.
TEST_CASE("EffectComposer: consecutive frames are identical", "[postprocessing]") {

    auto scene = platedScene(Color(0.6f, 0.3f, 0.9f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.addPass(std::make_shared<ShaderPass>(scaleShader(0.5f)));

    composer.render();
    const auto first = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    composer.render();
    const auto second = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    INFO("frame 1 " << first.r << "," << first.g << "," << first.b
                    << "  frame 2 " << second.r << "," << second.g << "," << second.b);
    CHECK(std::abs(second.r - first.r) < 2.0);
    CHECK(std::abs(second.g - first.g) < 2.0);
    CHECK(std::abs(second.b - first.b) < 2.0);
}

// The composer must survive being resized more than once: RenderTarget::setSize
// releases the backend's resources through the dispose event, and a latched
// dispose would leave the second resize rendering into framebuffers still sized
// for the first.
TEST_CASE("EffectComposer: survives repeated resizes", "[postprocessing]") {

    auto scene = platedScene(Color(0.4f, 0.4f, 0.4f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));

    composer.setSize(32, 32);
    composer.render();

    composer.setSize(RT_WIDTH, RT_HEIGHT);
    composer.render();

    CHECK(composer.readBuffer().width == RT_WIDTH);
    CHECK(composer.readBuffer().height == RT_HEIGHT);

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    // 0.4 -> byte 102, and a stale framebuffer would not fill the frame at all.
    INFO("expected ~102, got " << composed.r);
    CHECK(std::abs(composed.r - 102.0) < 3.0);
}

// Multisampled composer targets have to resolve into their texture before the
// next pass samples it. Without the resolve blit the chain reads an
// uninitialised texture — black, not the plate.
TEST_CASE("EffectComposer: multisampled targets resolve", "[postprocessing]") {

    auto scene = platedScene(Color(0.7f, 0.2f, 0.4f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer::Options options;
    options.samples = 4;

    EffectComposer composer(renderer, options);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.addPass(std::make_shared<ShaderPass>(scaleShader(1.f)));
    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    INFO("expected ~(179,51,102), got " << composed.r << "," << composed.g << "," << composed.b);
    CHECK(std::abs(composed.r - 179.0) < 3.0);
    CHECK(std::abs(composed.g - 51.0) < 3.0);
    CHECK(std::abs(composed.b - 102.0) < 3.0);
}

// MSAA is meant to be something the composer keeps, not something adding a
// composer costs you: a slanted edge through a multisampled chain must produce
// intermediate coverage pixels that a single-sampled chain does not.
TEST_CASE("EffectComposer: multisampled targets anti-alias edges", "[postprocessing]") {

    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);

    auto mat = MeshBasicMaterial::create();
    mat->color = Color(1, 1, 1);
    auto plate = Mesh::create(PlaneGeometry::create(1.2f, 1.2f), mat);
    plate->rotateZ(0.4f);// slanted edges, so coverage varies along them
    scene->add(plate);

    auto camera = plateCamera();

    const auto intermediateCount = [&](unsigned int samples) {
        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.setClearColor(Color(0, 0, 0));

        EffectComposer::Options options;
        options.samples = samples;

        EffectComposer composer(renderer, options);
        composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
        composer.render();

        return countIntermediatePixels(renderer.readRGBPixels());
    };

    const int aliased = intermediateCount(0);
    const int antialiased = intermediateCount(4);

    INFO("intermediate pixels: samples=0 " << aliased << ", samples=4 " << antialiased);
    CHECK(antialiased > aliased);
}

// A MaskPass confines the passes between it and ClearMaskPass to where the mask
// scene drew. Outside the mask the image has to survive untouched, which is
// what the composer's mid-chain copy under an active mask is for.
TEST_CASE("EffectComposer: MaskPass confines the passes it brackets", "[postprocessing]") {

    auto scene = platedScene(Color(0.8f, 0.8f, 0.8f));
    auto camera = plateCamera();

    // Mask covers the left half of the frame.
    auto maskScene = Scene::create();
    auto maskMat = MeshBasicMaterial::create();
    auto maskPlate = Mesh::create(PlaneGeometry::create(1, 4), maskMat);
    maskPlate->position.x = -0.5f;
    maskScene->add(maskPlate);

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.addPass(std::make_shared<MaskPass>(*maskScene, *camera));
    composer.addPass(std::make_shared<ShaderPass>(scaleShader(0.25f)));
    composer.addPass(std::make_shared<ClearMaskPass>());
    composer.render();

    const auto px = renderer.readRGBPixels();

    const auto sampleAt = [&](int x, int y) {
        return static_cast<double>(px[(static_cast<size_t>(y) * RT_WIDTH + x) * 3]);
    };

    const double masked = sampleAt(RT_WIDTH / 4, RT_HEIGHT / 2);       // inside the mask
    const double untouched = sampleAt(3 * RT_WIDTH / 4, RT_HEIGHT / 2);// outside it

    INFO("inside mask " << masked << " (expect ~51), outside " << untouched << " (expect ~204)");
    CHECK(std::abs(masked - 51.0) < 4.0);
    CHECK(std::abs(untouched - 204.0) < 4.0);
}
