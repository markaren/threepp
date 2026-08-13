// Regression: a closed CylinderGeometry cap must rasterize as a full disc.
// Reported from the ArduPilot SITL demo, whose cylinder landing pad rendered
// with one pie wedge missing near the theta=0 seam ("pac-man"), even though
// the geometry itself is watertight (see the welded-edge test in
// tests/geometries). This looks straight down at an unlit red cylinder and
// requires every angular sector of the cap to contain red pixels.

#include "gl_test_helpers.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Cylinder cap rasterizes every angular sector") {

    std::shared_ptr<Canvas> canvas;
    try {
        canvas = std::make_shared<Canvas>(Canvas::Parameters().size(64, 64).headless(true));
    } catch (const std::exception& e) {
        SKIP("no GL context: " << e.what());
    }

    GLRenderer renderer(*canvas);
    renderer.setClearColor(Color::black);

    Scene scene;
    auto mat = MeshBasicMaterial::create();// unlit: pixels are pass/fail
    mat->color = Color::red;
    auto cylinder = Mesh::create(CylinderGeometry::create(1.6f, 1.6f, 0.04f, 16), mat);
    scene.add(cylinder);

    OrthographicCamera camera(-2, 2, 2, -2, 0.1f, 10);
    camera.position.set(0, 5, 0);
    camera.lookAt({0, 0, 0});
    camera.updateMatrixWorld();

    renderer.render(scene, camera);
    const auto px = renderer.readRGBPixels();

    constexpr int w = 64, h = 64;
    REQUIRE(px.size() == static_cast<std::size_t>(w * h * 3));

    // 32 sectors of the rim band (radius 0.5..0.95 of the cap): each must have
    // at least one red pixel. 16 radial segments -> a missing wedge blanks two
    // adjacent sectors, far beyond rasterization jitter.
    int emptySectors = 0;
    for (int s = 0; s < 32; ++s) {
        const double a0 = (s + 0.25) * 2.0 * math::PI / 32;// sample inside the sector
        const double a1 = (s + 0.75) * 2.0 * math::PI / 32;
        bool hit = false;
        for (const double a : {a0, (a0 + a1) / 2, a1}) {
            for (const double r : {0.55, 0.7, 0.85}) {
                // Cap radius 1.6 maps to 1.6/2 * (w/2) = 0.4*w/2 pixels from center.
                const int x = static_cast<int>(w / 2 + std::cos(a) * r * 0.4 * w);
                const int y = static_cast<int>(h / 2 + std::sin(a) * r * 0.4 * h);
                if (x < 0 || x >= w || y < 0 || y >= h) continue;
                const int i = (y * w + x) * 3;
                if (px[i] > 128) hit = true;
            }
        }
        if (!hit) {
            ++emptySectors;
            UNSCOPED_INFO("sector " << s << " (theta ~" << (s * 11.25) << " deg) has no cap pixels");
        }
    }
    CHECK(emptySectors == 0);
}
