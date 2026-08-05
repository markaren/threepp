// The back-to-front sort inside SplatCloud, checked on the CPU.
//
// SplatCloud needs no GL context to be built or sorted: the textures are
// plain CPU buffers until something uploads them, and update() writes the
// draw order into instanceColor. So the ordering — the property a splat
// renderer lives or dies by — can be asserted directly, without a window,
// and without inferring it from pixels.
//
// GLSplatCloud_test carries the end-to-end version of the same claim.

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

using namespace threepp;

namespace {

    SplatData cloudOf(const std::vector<Vector3>& means) {

        SplatData data;
        data.resize(means.size(), 0);

        for (size_t i = 0; i < means.size(); ++i) {

            data.means[i] = means[i];
            data.scales[i].set(0.05f, 0.05f, 0.05f);
            data.rotations[i].set(0.f, 0.f, 0.f, 1.f);
            data.opacities[i] = 0.9f;
        }

        return data;
    }

    // The draw order update() produced: drawOrder[slot] = splat index.
    std::vector<size_t> drawOrder(SplatCloud& cloud) {

        const auto& slots = cloud.instanceColor()->array();

        std::vector<size_t> order;
        order.reserve(cloud.splatCount());
        for (size_t i = 0; i < cloud.splatCount(); ++i) {

            order.push_back(static_cast<size_t>(slots[i * 3] + 0.5f));
        }
        return order;
    }

    // Where splat `splat` was drawn. Later == nearer the camera.
    size_t slotOf(const std::vector<size_t>& order, size_t splat) {

        for (size_t i = 0; i < order.size(); ++i) {

            if (order[i] == splat) return i;
        }
        return order.size();
    }

}// namespace


TEST_CASE("SplatCloud: the draw order is back to front, from both sides") {

    auto cloud = SplatCloud::create(cloudOf({{0.f, 0.f, 1.f}, {0.f, 0.f, -1.f}}));

    auto front = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    front->position.set(0, 0, 5);
    front->lookAt(Vector3{0, 0, 0});
    cloud->update(*front);

    auto order = drawOrder(*cloud);
    CHECK(slotOf(order, 0) > slotOf(order, 1));// splat 0 is nearer, drawn last

    auto back = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    back->position.set(0, 0, -5);
    back->lookAt(Vector3{0, 0, 0});
    cloud->update(*back);

    order = drawOrder(*cloud);
    CHECK(slotOf(order, 1) > slotOf(order, 0));
}

TEST_CASE("SplatCloud: one far outlier does not coarsen the sort for everything else") {

    // The 16-bit key spread over min..max view depth is 0.06 units per bucket
    // once a single splat 4000 units away has set the range, and two splats
    // 0.006 apart then land in the SAME bucket. A stable counting sort keeps
    // ties in index order, so the pair comes out in file order from both
    // sides of the scene: right by luck from one, confidently wrong from the
    // other. Clamping the key range to a robust percentile is what fixes it,
    // and this test fails without that clamp.
    std::vector<Vector3> means;
    for (int i = 0; i < 200; ++i) {

        const float t = static_cast<float>(i) / 199.f;
        means.emplace_back((i % 2 ? -1.f : 1.f) * (1.2f + 0.6f * t), (t - 0.5f) * 2.f, t - 0.5f);
    }
    means.emplace_back(0.f, 0.f, -4000.f);// the stray that sets the range
    const size_t nearer = means.size();
    means.emplace_back(0.f, 0.f, 0.003f);
    const size_t farther = means.size();
    means.emplace_back(0.f, 0.f, -0.003f);

    auto cloud = SplatCloud::create(cloudOf(means));

    auto front = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    front->position.set(0, 0, 5);
    front->lookAt(Vector3{0, 0, 0});
    cloud->update(*front);

    auto order = drawOrder(*cloud);
    INFO("+z: nearer at slot " << slotOf(order, nearer) << ", farther at " << slotOf(order, farther));
    CHECK(slotOf(order, nearer) > slotOf(order, farther));

    // And from the other side, where the two swap roles. This is the half
    // that index order gets right by accident.
    auto back = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    back->position.set(0, 0, -5);
    back->lookAt(Vector3{0, 0, 0});
    cloud->update(*back);

    order = drawOrder(*cloud);
    INFO("-z: nearer at slot " << slotOf(order, nearer) << ", farther at " << slotOf(order, farther));
    CHECK(slotOf(order, farther) > slotOf(order, nearer));
}

TEST_CASE("SplatCloud: the sort is stable, so equal depths keep index order") {

    // Ties are what carried the popping-free result: splats at the same depth
    // must not shuffle between frames just because the camera moved a little.
    std::vector<Vector3> means;
    for (int i = 0; i < 64; ++i) means.emplace_back(0.02f * static_cast<float>(i), 0.f, 0.f);

    auto cloud = SplatCloud::create(cloudOf(means));

    auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    camera->position.set(0, 0, 5);
    camera->lookAt(Vector3{0, 0, 0});
    cloud->update(*camera);

    const auto order = drawOrder(*cloud);
    for (size_t i = 0; i < order.size(); ++i) CHECK(order[i] == i);
}

TEST_CASE("SplatCloud: a cloud at a single depth still sorts") {

    // The percentile interval collapses (p1 == p99) and the code has to fall
    // back rather than divide by nothing.
    std::vector<Vector3> means;
    for (int i = 0; i < 32; ++i) means.emplace_back(0.05f * static_cast<float>(i), 0.f, 0.f);

    auto cloud = SplatCloud::create(cloudOf(means));

    auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    camera->position.set(0, 0, 5);
    camera->lookAt(Vector3{0, 0, 0});

    REQUIRE_NOTHROW(cloud->update(*camera));
    CHECK(drawOrder(*cloud).size() == means.size());
}

TEST_CASE("SplatCloud: a NaN mean does not poison the key range") {

    // A corrupt mean must not decide the interval every other splat is
    // quantised against. The bad splat still gets a key; the shader is what
    // declines to draw it.
    std::vector<Vector3> means;
    for (int i = 0; i < 64; ++i) means.emplace_back(0.f, 0.f, -1.f + 2.f * static_cast<float>(i) / 63.f);
    means.emplace_back(0.f, 0.f, std::nanf(""));

    auto cloud = SplatCloud::create(cloudOf(means));

    auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 100);
    camera->position.set(0, 0, 5);
    camera->lookAt(Vector3{0, 0, 0});
    REQUIRE_NOTHROW(cloud->update(*camera));

    // The 64 healthy splats are still ordered back to front among themselves:
    // splat 0 is the farthest, splat 63 the nearest.
    const auto order = drawOrder(*cloud);
    CHECK(slotOf(order, 63) > slotOf(order, 0));
}
