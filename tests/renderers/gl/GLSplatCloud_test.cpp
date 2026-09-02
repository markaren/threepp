// End-to-end GL pixel tests for SplatCloud: does a Gaussian land where it
// should, in the colour it should, blended in the right order, and does a
// degenerate splat poison the frame.
//
// The ordering case is the one worth having. A splat cloud with a broken sort
// still renders — it renders confidently, and wrong — so the test drives the
// same two splats from both sides of the scene and demands the answer flip.

#include "gl_test_helpers.hpp"

#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"

#include <cmath>
#include <limits>
#include <catch2/catch_approx.hpp>

namespace {

    // A cloud of hand-placed, isotropic, flat-coloured Gaussians. Degree 0, so
    // colour is view-independent and a pixel check means what it says.
    struct SplatSpec {
        Vector3 position;
        Vector3 color;
        float scale;
        float opacity;
    };

    std::shared_ptr<SplatCloud> makeCloud(const std::vector<SplatSpec>& specs) {

        SplatData data;
        data.resize(specs.size(), 0);

        for (size_t i = 0; i < specs.size(); ++i) {

            data.means[i] = specs[i].position;
            data.scales[i].set(specs[i].scale, specs[i].scale, specs[i].scale);
            data.rotations[i].set(0.f, 0.f, 0.f, 1.f);
            data.opacities[i] = specs[i].opacity;
            data.setDcColor(i, specs[i].color);
        }

        return SplatCloud::create(std::move(data));
    }

    // Renders through a render target so the readback is linear — the splat
    // shader is a RawShaderMaterial and writes its colour out untouched, so a
    // red splat should arrive as a red pixel with no encode in between.
    std::vector<unsigned char> renderSplats(const std::shared_ptr<SplatCloud>& cloud,
                                            Camera& camera, const Color& clear,
                                            bool explicitUpdate = true) {

        auto scene = Scene::create();
        scene->add(cloud);

        // Explicit, before render(): the renderer uploads splatIndex while
        // building the render list, which is earlier than onBeforeRender, so
        // the fallback path inside the object would land the sort a frame late.
        //
        // explicitUpdate == false is the OTHER caller — an app that just adds a
        // cloud to a scene and renders, which is what the editor does. Everything
        // the shader binds has to already exist by then; see the case at the
        // bottom of this file.
        if (explicitUpdate) cloud->update(camera);

        GLRenderer renderer(glCanvas());
        renderer.setClearColor(clear);
        auto target = GLRenderTarget::create(RT_WIDTH, RT_HEIGHT, GLRenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, camera);
        auto pixels = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();

        // The cloud outlives this scene (the ordering test renders it twice).
        cloud->removeFromParent();

        return pixels;
    }

    AvgColor pixelAt(const std::vector<unsigned char>& px, int x, int y) {

        const int i = (y * RT_WIDTH + x) * 3;
        return {static_cast<double>(px[i]), static_cast<double>(px[i + 1]), static_cast<double>(px[i + 2])};
    }

    // Magenta is the shader's non-finite sentinel. Exactly (255, 0, 255).
    int countMagenta(const std::vector<unsigned char>& px) {

        int n = 0;
        for (size_t i = 0; i + 2 < px.size(); i += 3) {

            if (px[i] > 250 && px[i + 1] < 5 && px[i + 2] > 250) ++n;
        }
        return n;
    }

}// namespace


TEST_CASE("GL splats: a single Gaussian lands at the centre in its own colour") {

    auto cloud = makeCloud({{{0, 0, 0}, {1.f, 0.f, 0.f}, 0.25f, 1.0f}});

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    const auto centre = pixelAt(pixels, RT_WIDTH / 2, RT_HEIGHT / 2);
    INFO("centre = " << centre.r << ", " << centre.g << ", " << centre.b);
    CHECK(centre.r > 200.0);
    CHECK(centre.r > centre.g + 100.0);
    CHECK(centre.r > centre.b + 100.0);

    // ... and the corners are still background.
    const int corners[4][2] = {{0, 0}, {RT_WIDTH - 1, 0}, {0, RT_HEIGHT - 1}, {RT_WIDTH - 1, RT_HEIGHT - 1}};
    for (const auto& corner : corners) {

        const auto px = pixelAt(pixels, corner[0], corner[1]);
        INFO("corner " << corner[0] << "," << corner[1]);
        CHECK(px.r < 5.0);
        CHECK(px.g < 5.0);
        CHECK(px.b < 5.0);
    }
}

TEST_CASE("GL splats: the Gaussian falls off away from its centre") {

    auto cloud = makeCloud({{{0, 0, 0}, {1.f, 1.f, 1.f}, 0.3f, 1.0f}});

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000));

    const auto centre = pixelAt(pixels, RT_WIDTH / 2, RT_HEIGHT / 2);
    const auto inner = pixelAt(pixels, RT_WIDTH / 2 + 6, RT_HEIGHT / 2);
    const auto outer = pixelAt(pixels, RT_WIDTH / 2 + 12, RT_HEIGHT / 2);

    INFO("centre " << centre.r << ", +6px " << inner.r << ", +12px " << outer.r);
    CHECK(centre.r > inner.r);
    CHECK(inner.r > outer.r);
    // A hard-edged disc would fail this: the profile has to be smooth.
    CHECK(inner.r > 5.0);
}

TEST_CASE("GL splats: draw order follows the camera, from both sides") {

    // Red nearer +z, blue nearer -z, overlapping on screen. Index order puts
    // red first, so an unsorted cloud would answer "blue" from both sides.
    auto cloud = makeCloud({
            {{0, 0, 1.0f}, {1.f, 0.f, 0.f}, 0.30f, 0.95f}, // red
            {{0, 0, -1.0f}, {0.f, 0.f, 1.f}, 0.30f, 0.95f} // blue
    });

    auto front = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    front->position.set(0, 0, 5);
    front->lookAt(Vector3{0, 0, 0});

    const auto fromFront = renderSplats(cloud, *front, Color(0x000000));
    const auto f = pixelAt(fromFront, RT_WIDTH / 2, RT_HEIGHT / 2);
    INFO("from +z: " << f.r << ", " << f.g << ", " << f.b);
    CHECK(f.r > f.b);

    auto back = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    back->position.set(0, 0, -5);
    back->lookAt(Vector3{0, 0, 0});

    const auto fromBack = renderSplats(cloud, *back, Color(0x000000));
    const auto b = pixelAt(fromBack, RT_WIDTH / 2, RT_HEIGHT / 2);
    INFO("from -z: " << b.r << ", " << b.g << ", " << b.b);
    CHECK(b.b > b.r);
}

TEST_CASE("GL splats: one far outlier does not coarsen the sort for everything else") {

    // The 16-bit sort key is spread over the view-depth range of the cloud.
    // Let a single stray splat 4000 units away set that range and one bucket
    // is 0.06 units wide -- wider than the separation between two splats that
    // are genuinely in front of one another, which puts them in the SAME
    // bucket, which makes the sort fall back on index order for the pair. The
    // result is not noise: it is confidently wrong from one side and right
    // from the other, and it looks perfectly plausible either way.
    //
    // Clamping the key range to a robust percentile of the depths fixes it,
    // and this test is the reason to believe that: without the clamp the
    // "from +z" check below fails, because blue (the later index) wins the
    // tie from both sides.
    std::vector<SplatSpec> specs;

    // Enough splats for a percentile to mean something, kept well away from
    // the centre pixel and spread over a normal-looking depth range.
    for (int i = 0; i < 200; ++i) {

        const float t = static_cast<float>(i) / 199.f;
        const float side = (i % 2 == 0) ? 1.f : -1.f;
        specs.push_back({{side * (1.2f + 0.6f * t), (t - 0.5f) * 2.f, (t - 0.5f)},
                         {0.f, 0.35f, 0.f},
                         0.05f,
                         0.5f});
    }

    // The stray. Past the far plane, so it paints nothing and can only
    // influence the picture through the sort.
    specs.push_back({{0.f, 0.f, -4000.f}, {1.f, 1.f, 1.f}, 0.05f, 0.9f});

    // The pair: 0.006 apart, an order of magnitude inside the unclamped
    // bucket. Red is the lower index, so a tie is won by blue.
    specs.push_back({{0.f, 0.f, 0.003f}, {1.f, 0.f, 0.f}, 0.30f, 0.95f});
    specs.push_back({{0.f, 0.f, -0.003f}, {0.f, 0.f, 1.f}, 0.30f, 0.95f});

    auto cloud = makeCloud(specs);
    cloud->frustumCulled = false;

    auto front = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    front->position.set(0, 0, 5);
    front->lookAt(Vector3{0, 0, 0});

    const auto f = pixelAt(renderSplats(cloud, *front, Color(0x000000)), RT_WIDTH / 2, RT_HEIGHT / 2);
    INFO("from +z: " << f.r << ", " << f.g << ", " << f.b);
    CHECK(f.r > f.b);

    auto back = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    back->position.set(0, 0, -5);
    back->lookAt(Vector3{0, 0, 0});

    const auto b = pixelAt(renderSplats(cloud, *back, Color(0x000000)), RT_WIDTH / 2, RT_HEIGHT / 2);
    INFO("from -z: " << b.r << ", " << b.g << ", " << b.b);
    CHECK(b.b > b.r);
}

TEST_CASE("GL splats: opacity controls how much background survives") {

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    // Green splat over a red background. Straight alpha, so the centre pixel
    // is opacity*green + (1-opacity)*background.
    auto faint = makeCloud({{{0, 0, 0}, {0.f, 1.f, 0.f}, 0.25f, 0.25f}});
    auto solid = makeCloud({{{0, 0, 0}, {0.f, 1.f, 0.f}, 0.25f, 0.95f}});

    const Color background(1.f, 0.f, 0.f);

    const auto faintPixels = renderSplats(faint, *camera, background);
    const auto solidPixels = renderSplats(solid, *camera, background);

    const auto faintCentre = pixelAt(faintPixels, RT_WIDTH / 2, RT_HEIGHT / 2);
    const auto solidCentre = pixelAt(solidPixels, RT_WIDTH / 2, RT_HEIGHT / 2);

    INFO("faint " << faintCentre.r << "," << faintCentre.g
                  << "  solid " << solidCentre.r << "," << solidCentre.g);

    // Both blend; neither is opaque or invisible.
    CHECK(faintCentre.g > 20.0);
    CHECK(faintCentre.g < 120.0);
    CHECK(faintCentre.r > 120.0);// most of the red background still shows

    CHECK(solidCentre.g > 200.0);
    CHECK(solidCentre.r < 40.0);// the background is nearly gone

    CHECK(solidCentre.g > faintCentre.g);
}

TEST_CASE("GL splats: an orthographic camera does not shrink distant splats") {

    // The EWA Jacobian approximates the perspective divide. A parallel
    // projection has none, and using the perspective form anyway makes the far
    // splat smaller — which looks entirely plausible until you notice the
    // camera is orthographic.
    auto camera = OrthographicCamera::create(-2, 2, 2, -2, 0.1f, 100);
    camera->position.set(0, 0, 6);
    camera->lookAt(Vector3{0, 0, 0});

    auto nearCloud = makeCloud({{{0, 0, 2.f}, {1.f, 1.f, 1.f}, 0.3f, 0.9f}});
    auto farCloud = makeCloud({{{0, 0, -2.f}, {1.f, 1.f, 1.f}, 0.3f, 0.9f}});

    const int nearLit = countNonBlack(renderSplats(nearCloud, *camera, Color(0x000000)));
    const int farLit = countNonBlack(renderSplats(farCloud, *camera, Color(0x000000)));

    INFO("near covers " << nearLit << " px, far covers " << farLit << " px");
    REQUIRE(nearLit > 0);
    REQUIRE(farLit > 0);

    const double ratio = static_cast<double>(std::min(nearLit, farLit)) / std::max(nearLit, farLit);
    CHECK(ratio > 0.9);
}

TEST_CASE("GL splats: a splat behind the camera paints nothing") {

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    auto behind = makeCloud({{{0, 0, 20.f}, {1.f, 1.f, 1.f}, 0.3f, 1.0f}});
    behind->frustumCulled = false;// force it through, so the shader's cull is what is tested

    const auto pixels = renderSplats(behind, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) == 0);
}

TEST_CASE("GL splats: degenerate splats leave the frame finite and correct") {

    // A zero-scale splat has an exactly-zero 3D covariance; a zero-opacity one
    // contributes nothing; a behind-camera one must not project. All three sit
    // alongside a normal red splat, which must come out unharmed.
    // Green is the tell-tale: only the two splats that must contribute nothing
    // carry any, so a single green pixel anywhere means one of them leaked.
    auto cloud = makeCloud({
            {{-0.4f, 0.f, 0.f}, {0.f, 0.f, 1.f}, 0.f, 1.0f},   // zero scale
            {{0.4f, 0.f, 0.f}, {0.f, 1.f, 0.f}, 0.3f, 1e-6f},  // ~zero opacity
            {{0.f, 0.f, 40.f}, {0.f, 1.f, 1.f}, 0.3f, 1.0f},   // behind the camera
            {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, 0.25f, 1.0f}    // the good one
    });
    cloud->frustumCulled = false;
    cloud->setDebugNonFinite(true);// paint NaN magenta instead of hiding it

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000));
    REQUIRE(pixels.size() == DATA_SIZE);

    INFO("magenta pixels = " << countMagenta(pixels));
    CHECK(countMagenta(pixels) == 0);

    // The healthy splat is still there and still red.
    const auto centre = pixelAt(pixels, RT_WIDTH / 2, RT_HEIGHT / 2);
    INFO("centre = " << centre.r << ", " << centre.g << ", " << centre.b);
    CHECK(centre.r > 200.0);
    CHECK(centre.g < 40.0);

    // Neither green splat contributed anything, anywhere.
    int maxGreen = 0;
    for (size_t i = 1; i < pixels.size(); i += 3) maxGreen = std::max(maxGreen, static_cast<int>(pixels[i]));
    INFO("brightest green anywhere = " << maxGreen);
    CHECK(maxGreen < 5);
}

TEST_CASE("GL splats: the non-finite sentinel is not dead code") {

    // Every other test asserts the absence of magenta, which proves nothing
    // unless magenta can appear at all. A NaN in the SH coefficients is the
    // realistic way in — a corrupt file — and it reaches the fragment stage
    // through vColor with the geometry still perfectly finite.
    auto poison = [] {
        SplatData data;
        data.resize(1, 0);
        data.scales[0].set(0.25f, 0.25f, 0.25f);
        data.rotations[0].set(0.f, 0.f, 0.f, 1.f);
        data.opacities[0] = 1.f;
        data.sh[0] = std::numeric_limits<float>::quiet_NaN();
        return SplatCloud::create(std::move(data));
    };

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    auto loud = poison();
    loud->setDebugNonFinite(true);
    const auto shouted = renderSplats(loud, *camera, Color(0x000000));
    INFO("with the debug flag on, magenta pixels = " << countMagenta(shouted));
    CHECK(countMagenta(shouted) > 0);

    // With the flag off the same splat is silently dropped: a NaN never
    // reaches the framebuffer either way.
    auto quiet = poison();
    const auto hushed = renderSplats(quiet, *camera, Color(0x000000));
    CHECK(countMagenta(hushed) == 0);
    CHECK(countNonBlack(hushed) == 0);
}

TEST_CASE("GL splats: a procedurally generated cloud renders without NaN") {

    SplatGenerator::Options options;
    options.count = 400;
    options.shDegree = 3;
    options.seed = 99u;
    options.includeDegenerates = true;
    options.extent.set(3.f, 3.f, 3.f);

    auto cloud = SplatCloud::create(SplatGenerator::generate(options));
    cloud->setDebugNonFinite(true);

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 6);
    camera->lookAt(Vector3{0, 0, 0});

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000));

    CHECK(countMagenta(pixels) == 0);
    CHECK(countNonBlack(pixels) > PIXEL_COUNT / 8);
}

TEST_CASE("GL splats: an empty cloud renders nothing and does not crash") {

    auto cloud = SplatCloud::create(SplatData{});

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000));

    CHECK(countNonBlack(pixels) == 0);
}

TEST_CASE("GL splats: a cloud added to a scene draws without an explicit update") {

    // The editor's path, and the one every other case in this file skips by
    // calling update(camera) first. It used to throw rather than draw: the
    // sorted-index attribute was created lazily inside onBeforeRender, which
    // runs AFTER GLRenderer has walked the scene and decided which buffers to
    // upload, so the draw bound an attribute with no GL buffer behind it
    // ("invalid unordered_map key"). Every splat import in the editor died on
    // it, and no test here could see it.
    auto cloud = makeCloud({{{0, 0, 0}, {1.f, 0.f, 0.f}, 0.25f, 1.0f}});

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000), /*explicitUpdate*/ false);

    const auto centre = pixelAt(pixels, RT_WIDTH / 2, RT_HEIGHT / 2);
    CHECK(centre.r > 200.0);
    CHECK(centre.g < 40.0);
    CHECK(countMagenta(pixels) == 0);
}

TEST_CASE("GL splats: an empty cloud drawn without an explicit update is also fine") {

    // Same ordering, and the case that made the attribute allocation subtle:
    // an empty cloud has no instances to draw but the shader still binds
    // splatIndex, so the buffer has to exist for a draw that draws nothing.
    auto cloud = SplatCloud::create(SplatData{});

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);

    const auto pixels = renderSplats(cloud, *camera, Color(0x000000), /*explicitUpdate*/ false);

    CHECK(countNonBlack(pixels) == 0);
}


TEST_CASE("GL splats: point mix 1 draws a disc of the requested pixel size, not the Gaussian") {

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    // A wide Gaussian: at this framing sigma 0.5 is ~8.6 px, so a pixel 6 px
    // from the centre still gets alpha ~0.78 of the green in Gaussian mode.
    // As a 4 px disc (radius 2, feather to 2.5) that pixel is background.
    auto cloud = makeCloud({{{0, 0, 0}, {0.f, 1.f, 0.f}, 0.5f, 0.9f}});
    const Color background(1.f, 0.f, 0.f);
    const int cx = RT_WIDTH / 2, cy = RT_HEIGHT / 2;

    const auto gaussian = renderSplats(cloud, *camera, background);
    const auto gCentre = pixelAt(gaussian, cx, cy);
    const auto gSide = pixelAt(gaussian, cx + 6, cy);
    CHECK(gCentre.g > 200.0);
    CHECK(gSide.g > 100.0);// the Gaussian reaches out there

    cloud->setPointMix(1.f);
    cloud->setPointSize(4.f);
    CHECK(cloud->pointMix() == 1.f);
    CHECK(cloud->pointSize() == 4.f);
    CHECK(cloud->pointSigmaPixels() == Catch::Approx((2.f + 0.5f) / 3.f));

    const auto points = renderSplats(cloud, *camera, background);
    const auto pCentre = pixelAt(points, cx, cy);
    const auto pSide = pixelAt(points, cx + 6, cy);
    INFO("centre " << pCentre.g << " side " << pSide.g);
    // Opaque at the centre: opacity 0.9 lerps to 1 (clamped to 0.99 by the shader).
    CHECK(pCentre.g > 245.0);
    CHECK(pCentre.r < 10.0);
    // Background 6 px out: the disc ended at 2.5 px.
    CHECK(pSide.g < 5.0);
    CHECK(pSide.r > 250.0);

    // Mix 0 again is the Gaussian again.
    cloud->setPointMix(0.f);
    const auto back = renderSplats(cloud, *camera, background);
    CHECK(pixelAt(back, cx + 6, cy).g == Catch::Approx(gSide.g).margin(1.0));

    SECTION("the setters clamp") {

        cloud->setPointMix(3.f);
        CHECK(cloud->pointMix() == 1.f);
        cloud->setPointMix(-1.f);
        CHECK(cloud->pointMix() == 0.f);
        cloud->setPointSize(0.f);
        CHECK(cloud->pointSize() == 1.f);
        cloud->setPointSize(std::numeric_limits<float>::quiet_NaN());
        CHECK(cloud->pointSize() == 2.f);
    }
}

TEST_CASE("GL splats: at point mix 1 the nearest point wins, whatever its opacity") {

    auto camera = PerspectiveCamera::create(50, 1.0f, 0.1f, 100);
    camera->position.set(0, 0, 4);
    camera->lookAt(Vector3{0, 0, 0});

    // Two faint splats on the view axis, red in front of green. As Gaussians
    // the centre is a blend of both over the background; as points the front
    // one covers the back one outright.
    auto cloud = makeCloud({{{0, 0, 1.0f}, {1.f, 0.f, 0.f}, 0.2f, 0.3f},
                            {{0, 0, 0.0f}, {0.f, 1.f, 0.f}, 0.2f, 0.3f}});
    const Color background(0.f, 0.f, 1.f);
    const int cx = RT_WIDTH / 2, cy = RT_HEIGHT / 2;

    const auto blended = pixelAt(renderSplats(cloud, *camera, background), cx, cy);
    CHECK(blended.r > 40.0);
    CHECK(blended.g > 40.0);
    CHECK(blended.b > 40.0);

    cloud->setPointMix(1.f);
    cloud->setPointSize(6.f);
    const auto front = pixelAt(renderSplats(cloud, *camera, background), cx, cy);
    INFO("front " << front.r << "," << front.g << "," << front.b);
    CHECK(front.r > 245.0);
    CHECK(front.g < 8.0);
    CHECK(front.b < 8.0);

    // From the other side the green point is the near one.
    camera->position.set(0, 0, -4);
    camera->lookAt(Vector3{0, 0, 0});
    const auto behind = pixelAt(renderSplats(cloud, *camera, background), cx, cy);
    INFO("behind " << behind.r << "," << behind.g << "," << behind.b);
    CHECK(behind.g > 245.0);
    CHECK(behind.r < 8.0);
}
