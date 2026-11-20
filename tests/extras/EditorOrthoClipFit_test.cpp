// The far plane of the editor's orthographic viewport camera.
//
// It stood at ten kilometres for every scene. A parallel projection spends its
// depth buffer evenly, so that left steps of 0.6 mm: the print on a circuit
// board, 0.02 mm proud of the board, tied with it in depth, and looked at from
// straight above the whole of it went missing. The fit keeps the range as short
// as the scene allows.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/OrthoClipFit.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // The viewport's camera: `height` world units tall, standing `distance`
    // from `target` along `direction`.
    OrthographicCamera viewFrom(const Vector3& target, Vector3 direction, float distance, float height) {

        const float aspect = 16.f / 9.f;
        OrthographicCamera camera(-height * 0.5f * aspect, height * 0.5f * aspect, height * 0.5f, -height * 0.5f,
                                  0.05f, 10000.f);
        camera.position.copy(target).addScaledVector(direction.normalize(), distance);
        camera.lookAt(target);
        return camera;
    }

    float depthOf(const OrthographicCamera& camera, const Vector3& world) {

        Vector3 forward(0, 0, -1);
        forward.applyQuaternion(camera.quaternion);
        Vector3 offset;
        offset.subVectors(world, camera.position);
        return offset.dot(forward);
    }

    std::vector<Vector3> corners(const Box3& box) {

        std::vector<Vector3> out;
        for (const float x : {box.min().x, box.max().x})
            for (const float y : {box.min().y, box.max().y})
                for (const float z : {box.min().z, box.max().z}) out.emplace_back(x, y, z);
        return out;
    }

    // An Arduino Uno and a breadboard on a desk: what the fit was written for.
    const Box3 kBench(Vector3(-0.15f, 0.f, -0.1f), Vector3(0.15f, 0.02f, 0.1f));

    const std::vector<Vector3> kDirections{
            {0, 1, 1e-4f}, {0, -1, 1e-4f}, {0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {1, 1, 1}, {-2, 0.3f, 1}, {0.2f, 3, -1}};

}// namespace

TEST_CASE("OrthoClipFit: the far plane is behind everything in the bounds, from any side", "[editor][ortho]") {

    const Box3 yard(Vector3(-60, -2, -45), Vector3(80, 14, 30));

    for (const Box3& bounds : {kBench, yard}) {
        const Vector3 centre = bounds.getCenter();
        const float radius = bounds.getSize().length() * 0.5f;

        for (const Vector3& direction : kDirections) {
            // Standing clear of the scene, as the editor's views do, and looking
            // at something off its centre.
            const Vector3 target = centre + Vector3(radius * 0.3f, 0, -radius * 0.2f);
            const auto camera = viewFrom(target, direction, radius * 1.6f + 1.f, radius * 0.5f);

            const float farPlane = OrthoClipFit::farPlane(camera, bounds);
            for (const Vector3& corner : corners(bounds)) {
                CHECK(depthOf(camera, corner) < farPlane);
            }
        }
    }
}

TEST_CASE("OrthoClipFit: a small scene gets a short range", "[editor][ortho]") {

    for (const Vector3& direction : kDirections) {
        const auto camera = viewFrom(kBench.getCenter(), direction, 1.5f, 0.2f);

        const float range = OrthoClipFit::farPlane(camera, kBench) - camera.nearPlane;

        // 24 bits over the range: the step has to be far below the 0.02 mm a
        // board's print stands proud of it. Ten kilometres gave 0.6 mm.
        const float step = range / 16777216.f;
        CHECK(step < 1e-6f);
        CHECK(range < 4.f);
    }
}

TEST_CASE("OrthoClipFit: a large scene is not cut short, and is not padded to many times its depth", "[editor][ortho]") {

    const Box3 terrain(Vector3(-9000, -200, -9000), Vector3(9000, 900, 9000));
    const float radius = terrain.getSize().length() * 0.5f;
    const auto camera = viewFrom(terrain.getCenter(), {1, 1, 1}, radius + 1.f, 500.f);

    const float farPlane = OrthoClipFit::farPlane(camera, terrain);
    CHECK(farPlane > 2.f * radius);
    CHECK(farPlane < 2.f * radius * 1.5f + 600.f);
}

TEST_CASE("OrthoClipFit: the grid is held as far as the view sees it", "[editor][ortho]") {

    OrthoClipFit::Sheet grid;
    grid.radius = 20.f * std::sqrt(2.f);

    SECTION("from above, the grid's plane is inside the range and the range stays short") {
        const auto camera = viewFrom(kBench.getCenter(), {0, 1, 1e-4f}, 1.5f, 0.2f);
        const float farPlane = OrthoClipFit::farPlane(camera, kBench, &grid);

        CHECK(depthOf(camera, grid.centre) < farPlane);
        CHECK(farPlane - camera.nearPlane < 4.f);
    }

    SECTION("tilted, every grid point under the frustum is inside the range") {
        const auto camera = viewFrom(kBench.getCenter(), {1, 0.5f, 1}, 1.5f, 0.2f);
        const float farPlane = OrthoClipFit::farPlane(camera, kBench, &grid);

        Vector3 right(1, 0, 0), up(0, 1, 0), forward(0, 0, -1);
        right.applyQuaternion(camera.quaternion);
        up.applyQuaternion(camera.quaternion);
        forward.applyQuaternion(camera.quaternion);
        for (const float sx : {-1.f, 0.f, 1.f}) {
            for (const float sy : {-1.f, 0.f, 1.f}) {
                // Where the ray through this point of the frustum meets y = 0.
                Vector3 origin = camera.position;
                origin.addScaledVector(right, sx * camera.right).addScaledVector(up, sy * camera.top);
                const float along = -origin.y / forward.y;
                CHECK(along < farPlane);
            }
        }
        // A tilted look at a desk does not take the whole 40-unit grid with it.
        CHECK(farPlane - camera.nearPlane < 6.f);
    }

    SECTION("edge on, the grid's own rim bounds the range") {
        const auto camera = viewFrom(Vector3(0, 0.01f, 0), {0, 0, 1}, 1.5f, 0.2f);
        const float farPlane = OrthoClipFit::farPlane(camera, kBench, &grid);

        CHECK(farPlane > 1.5f + grid.radius);
        CHECK(farPlane < (1.5f + grid.radius) * 1.5f + 1.f);
    }

    SECTION("stood up into an axis view's plane, it costs no range") {
        grid.normal.set(0, 0, 1);
        const auto camera = viewFrom(kBench.getCenter(), {0, 0, 1}, 1.5f, 0.2f);
        const float farPlane = OrthoClipFit::farPlane(camera, kBench, &grid);

        CHECK(depthOf(camera, grid.centre) < farPlane);
        CHECK(farPlane - camera.nearPlane < 4.f);
    }
}

TEST_CASE("OrthoClipFit: the answer moves in steps", "[editor][ortho]") {

    // Orbiting about the scene: the projection must not change every frame.
    int changes = 0;
    float last = 0.f;
    for (int i = 0; i < 360; ++i) {
        const float a = math::degToRad(static_cast<float>(i));
        const auto camera = viewFrom(kBench.getCenter() + Vector3(0.05f, 0, 0.02f),
                                     {std::cos(a), 0.6f, std::sin(a)}, 1.5f, 0.2f);
        const float farPlane = OrthoClipFit::farPlane(camera, kBench);
        if (i > 0 && farPlane != last) ++changes;
        last = farPlane;
    }
    CHECK(changes <= 4);
}

TEST_CASE("OrthoClipFit: an empty scene still gets a range", "[editor][ortho]") {

    const auto camera = viewFrom({0, 0, 0}, {1, 1, 1}, 5.f, 2.f);

    const float farPlane = OrthoClipFit::farPlane(camera, Box3());
    CHECK(farPlane >= camera.nearPlane + 1.f);
    CHECK(std::isfinite(farPlane));
}
