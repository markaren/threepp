// The environment background under an ORTHOGRAPHIC camera.
//
// GLBackground draws a cube-texture background as a unit box centred on the
// camera (the three.js trick). Under perspective that fills the view at any
// size: the eye is inside the box and the perspective divide expands whatever
// it hits, so the box's actual size never matters. A parallel projection has no
// divide, so the same box projects at its literal one unit and the environment
// appears as a small box in the middle of the viewport — which is what this
// backend used to do, and what three.js still does (its WebGLBackground never
// scales the box and carries no orthographic branch).
//
// The contract asserted here is the one the Vulkan backend already states in
// camera_ray.glsl: a parallel camera has ONE view direction for every pixel.
// So an orthographic viewport must be filled edge to edge, and filled with a
// SINGLE face of the cube — the one the camera faces. Both halves matter: a box
// that were merely scaled up would satisfy the coverage claim while still
// showing several faces meeting mid-screen, so uniformity is what pins the
// direction rather than just the size.

#include "gl_test_helpers.hpp"

#include "threepp/textures/CubeTexture.hpp"

namespace {

    // Six 1x1 faces, each a distinct colour, so the colour that comes back
    // names the face that was sampled. three.js cube face order: +X, -X, +Y,
    // -Y, +Z, -Z. RGBA (and the format set to match) because a 1x1 RGB face is
    // 3 bytes per row and would trip GL's 4-byte unpack alignment.
    std::shared_ptr<CubeTexture> makeFaceColouredCube() {
        auto face = [](unsigned char r, unsigned char g, unsigned char b) {
            return Image(std::vector<unsigned char>{r, g, b, 255}, 1, 1);
        };
        auto tex = CubeTexture::create({
                face(255, 0, 0),    // +X red
                face(0, 255, 0),    // -X green
                face(0, 0, 255),    // +Y blue
                face(255, 255, 0),  // -Y yellow
                face(255, 0, 255),  // +Z magenta
                face(0, 255, 255),  // -Z cyan
        });
        tex->format = Format::RGBA;
        return tex;
    }

    // The clear colour is deliberately one no cube face uses: any pixel still
    // wearing it is a pixel the background failed to cover.
    const Color kClear{0x101010};

}// namespace

TEST_CASE("orthographic camera fills the viewport with the environment", "[gl][background]") {

    Scene scene;
    scene.background = Background(makeFaceColouredCube());

    // Looking down -Z from +Z: the camera faces the -Z cube face (cyan). The
    // frustum is many times the unit box, which is what used to leave the
    // background as a small patch surrounded by clear colour.
    OrthographicCamera camera(-10, 10, 10, -10, 0.1f, 100);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);

    const auto pixels = renderWithGL(scene, camera, kClear);
    REQUIRE(pixels.size() == DATA_SIZE);

    SECTION("no pixel is left showing the clear colour") {
        // 0x10 = 16 on every channel. A generous margin still separates it from
        // any cube face, all of which have a channel at 0 and one at 255.
        int clearPixels = 0;
        for (int i = 0; i < PIXEL_COUNT; ++i) {
            const int r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2];
            if (r < 40 && g < 40 && b < 40) ++clearPixels;
        }
        INFO("pixels still showing the clear colour: " << clearPixels << " / " << PIXEL_COUNT);
        REQUIRE(clearPixels == 0);
    }

    SECTION("the whole viewport samples one direction, not several faces") {
        // One direction for every pixel ⇒ one colour for every pixel. A scaled
        // box would still show the seams where faces meet, and those seams are
        // exactly what this variance would catch.
        INFO("brightness variance across the viewport: " << brightnessVariance(pixels));
        REQUIRE(brightnessVariance(pixels) < 1.0);
    }

    SECTION("that direction is the camera's forward") {
        // Facing -Z ⇒ the -Z face ⇒ cyan (0, 255, 255).
        const auto avg = averageColor(pixels);
        INFO("average colour: " << avg.r << ", " << avg.g << ", " << avg.b);
        REQUIRE(avg.r < 40);
        REQUIRE(avg.g > 200);
        REQUIRE(avg.b > 200);
    }
}

TEST_CASE("turning the orthographic camera selects the face it faces", "[gl][background]") {

    Scene scene;
    scene.background = Background(makeFaceColouredCube());

    // Now looking down -X from +X. Pins that the direction tracks the camera
    // rather than being hard-coded, and that a rotated camera is still covered
    // edge to edge — the box is axis-aligned in world space, so its coverage has
    // to survive rotation.
    //
    // The face is the +X one (RED), not -X, and that is correct. Cube maps are
    // specified in a LEFT-handed coordinate system (the RenderMan convention GL
    // inherited); three.js and threepp are right-handed. The conversion is a
    // negation of x at sample time, which is what makes a loaded cube map appear
    // UNMIRRORED — and necessarily swaps which slot the ±X directions land in.
    // three.js says so itself in CubeTexture: "environment maps used in three.js
    // appear to have px and nx swapped". Getting the unmirrored image matters;
    // the label swap is the price, and it is r129's price too.
    //
    // This case therefore pins the SIGN of flipEnvMap, which is a float
    // multiplier and not a flag: +1 would show green here, and 0 — what writing
    // a bool `false` uploads — would collapse the sampled direction onto the yz
    // plane and show a seam rather than one flat colour.
    auto faceUnderCamera = [&](const Vector3& eye) {
        OrthographicCamera camera(-10, 10, 10, -10, 0.1f, 100);
        camera.position.copy(eye);
        camera.lookAt(0, 0, 0);

        const auto pixels = renderWithGL(scene, camera, kClear);
        REQUIRE(pixels.size() == DATA_SIZE);

        int clearPixels = 0;
        for (int i = 0; i < PIXEL_COUNT; ++i) {
            const int r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2];
            if (r < 40 && g < 40 && b < 40) ++clearPixels;
        }
        INFO("pixels still showing the clear colour: " << clearPixels << " / " << PIXEL_COUNT);
        REQUIRE(clearPixels == 0);

        // One flat colour: excludes a collapsed sample direction.
        INFO("brightness variance: " << brightnessVariance(pixels));
        REQUIRE(brightnessVariance(pixels) < 1.0);

        return averageColor(pixels);
    };

    {
        const auto avg = faceUnderCamera({5, 0, 0});// viewing along -X
        INFO("looking along -X, average colour: " << avg.r << ", " << avg.g << ", " << avg.b);
        REQUIRE(avg.r > 200);
        REQUIRE(avg.g < 40);
        REQUIRE(avg.b < 40);
    }
    {
        const auto avg = faceUnderCamera({-5, 0, 0});// viewing along +X
        INFO("looking along +X, average colour: " << avg.r << ", " << avg.g << ", " << avg.b);
        REQUIRE(avg.g > 200);
        REQUIRE(avg.r < 40);
        REQUIRE(avg.b < 40);
    }
}

TEST_CASE("a perspective camera still sees the environment on every side", "[gl][background]") {

    // The guard on the change: perspective goes down the untouched branch
    // (orthoDirection.w == 0 ⇒ the original per-vertex direction), so the box
    // still surrounds the eye and different parts of the view see different
    // faces. Uniformity here would mean the ortho path had leaked into it.
    Scene scene;
    scene.background = Background(makeFaceColouredCube());

    PerspectiveCamera camera(90, 1, 0.1f, 100);
    camera.position.set(0, 0, 0);
    camera.lookAt(1, 1, 0);// off-axis, so several faces are in frame

    const auto pixels = renderWithGL(scene, camera, kClear);
    REQUIRE(pixels.size() == DATA_SIZE);

    int clearPixels = 0;
    for (int i = 0; i < PIXEL_COUNT; ++i) {
        const int r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2];
        if (r < 40 && g < 40 && b < 40) ++clearPixels;
    }
    INFO("pixels still showing the clear colour: " << clearPixels << " / " << PIXEL_COUNT);
    REQUIRE(clearPixels == 0);

    INFO("brightness variance across the viewport: " << brightnessVariance(pixels));
    REQUIRE(brightnessVariance(pixels) > 1.0);
}
