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

#include <cmath>
#include <optional>
#include <set>
#include <utility>

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

// Geometry is encoded by the fragment shader, which threepp compiles against
// the bound target — but a clear bypasses the shader, so the clear colour is
// encoded on the CPU instead, and that encode has to read the same target. It
// did not: it read the renderer's output space unconditionally, so the
// background went into the linear intermediate already sRGB-encoded and the
// composer's output draw encoded it a second time. Every other test in this
// file clears to black, which is a fixed point of the encode and hides it
// completely; a dark non-black background is where it shows, and it shows
// large — 0x05050a arrived as (38,38,56) instead of (5,5,10).
TEST_CASE("EffectComposer: the scene background survives the chain", "[postprocessing]") {

    auto scene = Scene::create();
    scene->background = Color(0.02f, 0.02f, 0.04f);// dark, where the encode error is biggest
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::sRGB;

    renderer.render(*scene, *camera);
    const auto direct = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    // sRGB encode of linear 0.02 is 0.1544 -> byte 39. Encoding twice lands at
    // 106, which is what the composer used to show.
    INFO("direct " << direct.r << ", composed " << composed.r << " (expected ~39, twice-encoded ~106)");
    CHECK(std::abs(direct.r - 39.0) < 3.0);
    CHECK(std::abs(composed.r - direct.r) < 3.0);
}

// The renderer's own clear colour takes the same path as a scene background,
// and RenderPass clears with it immediately after binding the composer's
// target — before anything re-encodes for that bind.
TEST_CASE("EffectComposer: the renderer clear colour survives the chain", "[postprocessing]") {

    auto scene = Scene::create();
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::sRGB;
    renderer.setClearColor(Color(0.02f, 0.02f, 0.04f), 1.f);

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.render();

    const auto composed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    INFO("expected ~39, got " << composed.r);
    CHECK(std::abs(composed.r - 39.0) < 3.0);
}

// The intermediates hold linear light, so their precision is what a dark ramp
// has to survive. In 8 bits it does not: the output encode maps the first byte
// steps to 0, 13, 22, and a smooth gradient arrives as a handful of flat
// terraces. Measured as the number of distinct output values across a ramp
// that spans the dark end, where byte targets have almost nothing to spend.
TEST_CASE("EffectComposer: a dark gradient does not band", "[postprocessing]") {

    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);
    auto camera = plateCamera();

    // A horizontal ramp over the bottom 4% of the linear range.
    auto ramp = std::make_shared<ShaderPass>(Shader{
            UniformMap{{"tDiffuse", Uniform()}},
            R"(
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
            })",
            R"(
            uniform sampler2D tDiffuse;
            varying vec2 vUv;
            void main() {
                gl_FragColor = vec4( vec3( vUv.x * 0.04 ), 1.0 );
            })"});

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::sRGB;

    const auto levels = [&](const std::optional<Type>& type) {
        EffectComposer::Options options;
        options.type = type;

        EffectComposer composer(renderer, options);
        composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
        composer.addPass(ramp);
        composer.render();

        const auto px = renderer.readRGBPixels();
        std::set<unsigned char> seen;
        const int row = RT_HEIGHT / 2;
        for (int x = 0; x < RT_WIDTH; ++x) seen.insert(px[(row * RT_WIDTH + x) * 3]);

        return seen.size();
    };

    const auto banded = levels(Type::UnsignedByte);
    const auto smooth = levels(std::nullopt);// the default: half float

    // 64 pixels of ramp. Bytes can only reach the few linear codes below 0.04,
    // so they collapse to a handful of terraces; half float keeps most of them.
    INFO("byte targets " << banded << " levels, default " << smooth << " (of " << RT_WIDTH << " pixels)");
    CHECK(banded <= 12);
    CHECK(smooth > banded * 2);
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

// =============================================================================
// UnrealBloomPass
// =============================================================================

namespace {

    // A small bright square on black, so "did light spread beyond the source"
    // is a question about specific pixels.
    std::shared_ptr<Scene> emitterScene(float brightness) {

        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        auto mat = MeshBasicMaterial::create();
        mat->color = Color(brightness, brightness, brightness);

        scene->add(Mesh::create(PlaneGeometry::create(0.5f, 0.5f), mat));

        return scene;
    }

    // Mean brightness of a ring of pixels well outside the emitter - where
    // bloom shows up and nothing else does.
    double haloBrightness(const std::vector<unsigned char>& px) {

        double sum = 0;
        int count = 0;

        for (int y = 0; y < RT_HEIGHT; y++) {
            for (int x = 0; x < RT_WIDTH; x++) {
                const double dx = x - RT_WIDTH / 2.0;
                const double dy = y - RT_HEIGHT / 2.0;
                const double r = std::sqrt(dx * dx + dy * dy);
                if (r < 14.0 || r > 24.0) continue;// the emitter is ~8px half-width

                const size_t i = (static_cast<size_t>(y) * RT_WIDTH + x) * 3;
                sum += (px[i] + px[i + 1] + px[i + 2]) / 3.0;
                count++;
            }
        }

        return count > 0 ? sum / count : 0.0;
    }

}// namespace

// The whole point of the pass: light from a bright source has to appear where
// the source is not. A chain that ran the mips but never added them back, or
// composited black, leaves the halo at zero.
TEST_CASE("UnrealBloomPass: a bright source spreads light beyond itself", "[postprocessing]") {

    auto scene = emitterScene(1.f);
    auto camera = plateCamera();

    const auto haloWith = [&](float strength) {
        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.setClearColor(Color(0, 0, 0));

        EffectComposer composer(renderer);
        composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
        composer.addPass(std::make_shared<UnrealBloomPass>(Vector2(RT_WIDTH, RT_HEIGHT), strength, 0.5f, 0.2f));
        composer.render();

        return haloBrightness(renderer.readRGBPixels());
    };

    const double none = haloWith(0.f);
    const double bloomed = haloWith(3.f);

    INFO("halo without bloom " << none << ", with bloom " << bloomed);
    CHECK(none < 1.0);
    CHECK(bloomed > 8.0);
}

// Strength 0 has to be the identity. It is the knob reached for to A/B the
// effect, so a pass that still tints or dims at zero makes that comparison a
// lie - and would mean the blend is not actually additive.
TEST_CASE("UnrealBloomPass: strength 0 leaves the image untouched", "[postprocessing]") {

    auto scene = platedScene(Color(0.5f, 0.5f, 0.5f));
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.render();
    const auto plain = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    EffectComposer bloomComposer(renderer);
    bloomComposer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    bloomComposer.addPass(std::make_shared<UnrealBloomPass>(Vector2(RT_WIDTH, RT_HEIGHT), 0.f, 0.5f, 0.5f));
    bloomComposer.render();
    const auto bloomed = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);

    INFO("plain " << plain.r << ", strength-0 bloom " << bloomed.r);
    CHECK(std::abs(bloomed.r - plain.r) < 2.0);
}

// The threshold has to gate. A source below it must not bloom at all -
// otherwise the high pass is doing nothing and every mid-grey surface glows.
TEST_CASE("UnrealBloomPass: a source below the threshold does not bloom", "[postprocessing]") {

    auto scene = emitterScene(0.3f);
    auto camera = plateCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer composer(renderer);
    composer.addPass(std::make_shared<RenderPass>(*scene, *camera));
    composer.addPass(std::make_shared<UnrealBloomPass>(Vector2(RT_WIDTH, RT_HEIGHT), 3.f, 0.5f, 0.8f));
    composer.render();

    const double halo = haloBrightness(renderer.readRGBPixels());

    INFO("halo from a 0.3 source under a 0.8 threshold: " << halo);
    CHECK(halo < 1.0);
}

// =============================================================================
// BokehPass
// =============================================================================

namespace {

    // Two plates at different depths, one on each side of the frame, so one can
    // be in focus while the other is not.
    struct DepthScene {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
    };

    DepthScene twoPlateScene() {

        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);

        const auto plate = [](float z, float x, float size) {
            auto mat = MeshBasicMaterial::create();
            mat->color = Color(1, 1, 1);
            auto mesh = Mesh::create(PlaneGeometry::create(size, size * 8.f), mat);
            mesh->position.set(x, 0, z);
            return mesh;
        };

        scene->add(plate(-2.f, -0.7f, 0.6f)); // near, left half
        scene->add(plate(-12.f, 4.2f, 3.6f));// far, right half

        auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 50.f);
        camera->position.set(0, 0, 0);
        camera->lookAt(Vector3{0, 0, -1});

        return {scene, camera};
    }

    // Sum of squared horizontal gradients over a column band: high when an edge
    // in that band is crisp, low once it has been smeared.
    double gradientEnergy(const std::vector<unsigned char>& px, int x0, int x1) {

        double energy = 0;
        for (int y = 0; y < RT_HEIGHT; y++) {
            for (int x = x0; x < x1 - 1; x++) {
                const size_t i = (static_cast<size_t>(y) * RT_WIDTH + x) * 3;
                const double a = (px[i] + px[i + 1] + px[i + 2]) / 3.0;
                const double b = (px[i + 3] + px[i + 4] + px[i + 5]) / 3.0;
                energy += (b - a) * (b - a);
            }
        }
        return energy;
    }

}// namespace

// Focus has to select by depth: pulling focus from the near plate to the far
// one must sharpen the far edge and soften the near one. A pass blurring
// uniformly - reading a broken depth target, say - moves both together.
TEST_CASE("BokehPass: focus selects which depth stays sharp", "[postprocessing]") {

    auto depthScene = twoPlateScene();

    const auto energiesAt = [&](float focus) {
        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.setClearColor(Color(0, 0, 0));

        EffectComposer composer(renderer);
        composer.addPass(std::make_shared<RenderPass>(*depthScene.scene, *depthScene.camera));
        composer.addPass(std::make_shared<BokehPass>(*depthScene.scene, *depthScene.camera, focus, 0.05f, 0.02f));
        composer.render();

        const auto px = renderer.readRGBPixels();
        return std::pair<double, double>{gradientEnergy(px, 0, RT_WIDTH / 2),
                                         gradientEnergy(px, RT_WIDTH / 2, RT_WIDTH)};
    };

    const auto nearFocus = energiesAt(2.f); // focus the near plate
    const auto farFocus = energiesAt(12.f); // focus the far plate

    INFO("focus near: left " << nearFocus.first << " right " << nearFocus.second
                             << " | focus far: left " << farFocus.first << " right " << farFocus.second);

    // The near plate (left) is crisper when focused near; the far plate
    // (right) is crisper when focused far.
    CHECK(nearFocus.first > farFocus.first);
    CHECK(farFocus.second > nearFocus.second);
}

// Aperture 0 is the pinhole case and must be the identity, for the same reason
// strength 0 must be for bloom.
TEST_CASE("BokehPass: aperture 0 leaves the image untouched", "[postprocessing]") {

    auto depthScene = twoPlateScene();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.setClearColor(Color(0, 0, 0));

    EffectComposer plainComposer(renderer);
    plainComposer.addPass(std::make_shared<RenderPass>(*depthScene.scene, *depthScene.camera));
    plainComposer.render();
    const double plain = gradientEnergy(renderer.readRGBPixels(), 0, RT_WIDTH);

    EffectComposer bokehComposer(renderer);
    bokehComposer.addPass(std::make_shared<RenderPass>(*depthScene.scene, *depthScene.camera));
    bokehComposer.addPass(std::make_shared<BokehPass>(*depthScene.scene, *depthScene.camera, 5.f, 0.f, 0.02f));
    bokehComposer.render();
    const double pinhole = gradientEnergy(renderer.readRGBPixels(), 0, RT_WIDTH);

    INFO("plain " << plain << ", aperture-0 bokeh " << pinhole);
    CHECK(std::abs(pinhole - plain) < plain * 0.02);
}
