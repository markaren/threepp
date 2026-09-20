// A geometry group whose materialIndex runs past the material list.
//
// r129 reads `material[ group.materialIndex ]`, which is undefined for an
// out-of-range index, and skips that group on the very next line:
// `if ( groupMaterial && groupMaterial.visible )`. threepp ported the guard but
// reached the material with `.at()`, which throws std::out_of_range before the
// guard can do anything — so the guard could never fire, and a mesh with more
// groups than materials took the whole render down rather than drawing the
// groups that do have one.
//
// That combination is routine: imported models often carry more groups than the
// materials that survived conversion, and any code that adds a group before
// assigning its material passes through the same state for a frame.

#include "gl_test_helpers.hpp"

namespace {

    // Two quads side by side as one geometry, in two groups. The second group
    // deliberately points at a material index that does not exist.
    std::shared_ptr<BufferGeometry> twoGroups(unsigned secondIndex) {
        std::vector<float> pos{
                // left quad
                -1.5f, -0.6f, 0.f, -0.3f, -0.6f, 0.f, -0.3f, 0.6f, 0.f,
                -1.5f, -0.6f, 0.f, -0.3f, 0.6f, 0.f, -1.5f, 0.6f, 0.f,
                // right quad
                0.3f, -0.6f, 0.f, 1.5f, -0.6f, 0.f, 1.5f, 0.6f, 0.f,
                0.3f, -0.6f, 0.f, 1.5f, 0.6f, 0.f, 0.3f, 0.6f, 0.f};

        auto geo = BufferGeometry::create();
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->addGroup(0, 6, 0);
        geo->addGroup(6, 6, secondIndex);
        return geo;
    }

    std::vector<unsigned char> renderGroups(unsigned secondIndex, size_t materialCount) {
        auto scene = Scene::create();

        std::vector<std::shared_ptr<Material>> mats;
        auto red = MeshBasicMaterial::create();
        red->color = Color(0xff0000);
        mats.push_back(red);
        if (materialCount > 1) {
            auto green = MeshBasicMaterial::create();
            green->color = Color(0x00ff00);
            mats.push_back(green);
        }

        scene->add(Mesh::create(twoGroups(secondIndex), mats));

        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 3.f);
        camera.lookAt({0, 0, 0});

        return renderWithGL(*scene, camera, Color(0x000000));
    }

}// namespace

TEST_CASE("a group pointing past the material list is skipped, not fatal") {

    // Two materials, but the second group asks for index 5.
    std::vector<unsigned char> px;
    REQUIRE_NOTHROW(px = renderGroups(5, 2));
    REQUIRE(px.size() == DATA_SIZE);

    // The in-range group still drew: red on the left, nothing on the right.
    int redPixels = 0, greenPixels = 0;
    for (int i = 0; i < PIXEL_COUNT; ++i) {
        const int r = px[i * 3], g = px[i * 3 + 1];
        if (r > 100 && g < 60) ++redPixels;
        if (g > 100 && r < 60) ++greenPixels;
    }
    INFO("red pixels " << redPixels << ", green pixels " << greenPixels);
    CHECK(redPixels > 100);
    CHECK(greenPixels == 0);
}

TEST_CASE("groups still map to their materials when the indices are in range") {

    const auto px = renderGroups(1, 2);
    REQUIRE(px.size() == DATA_SIZE);

    int redPixels = 0, greenPixels = 0;
    for (int i = 0; i < PIXEL_COUNT; ++i) {
        const int r = px[i * 3], g = px[i * 3 + 1];
        if (r > 100 && g < 60) ++redPixels;
        if (g > 100 && r < 60) ++greenPixels;
    }
    INFO("red pixels " << redPixels << ", green pixels " << greenPixels);
    CHECK(redPixels > 100);
    CHECK(greenPixels > 100);
}
