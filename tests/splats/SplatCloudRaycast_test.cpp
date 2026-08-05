// Picking a splat cloud.
//
// The override exists because the inherited one is actively wrong here:
// InstancedMesh::raycast walks every instance's instanceMatrix, and SplatCloud
// leaves all of them at identity on purpose (the per-splat data lives in
// textures). So the base class tests the same unit quad at the origin N times
// and reports hits that have nothing to do with where the splats are.
//
// These assertions are about the SPHERE — the 3-sigma bound the constructor
// already computes for frustum culling — not about individual splats. Hitting
// a particular splat is a later refinement; being selectable at all is what an
// editor needs.

#include "threepp/core/Raycaster.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

using namespace threepp;

namespace {

    // A compact cloud centred on `centre`, small enough that its 3-sigma bound
    // is roughly the extent of the means themselves.
    SplatData cloudAt(const Vector3& centre, float halfExtent = 1.f) {

        const std::vector<Vector3> offsets{
                {-halfExtent, 0.f, 0.f}, {halfExtent, 0.f, 0.f}, {0.f, -halfExtent, 0.f}, {0.f, halfExtent, 0.f}, {0.f, 0.f, -halfExtent}, {0.f, 0.f, halfExtent}};

        SplatData data;
        data.resize(offsets.size(), 0);
        for (std::size_t i = 0; i < offsets.size(); ++i) {
            data.means[i].copy(centre).add(offsets[i]);
            data.scales[i].set(0.02f, 0.02f, 0.02f);
            data.rotations[i].set(0.f, 0.f, 0.f, 1.f);
            data.opacities[i] = 0.9f;
        }
        return data;
    }

}// namespace


TEST_CASE("a ray through the cloud hits it once", "[splats]") {

    auto cloud = SplatCloud::create(cloudAt({0.f, 0.f, 0.f}));
    cloud->updateMatrixWorld();

    Raycaster raycaster;
    raycaster.set({0.f, 0.f, 10.f}, {0.f, 0.f, -1.f});

    std::vector<Intersection> hits;
    cloud->raycast(raycaster, hits);

    // One hit, not one per splat: the whole cloud is the pickable thing.
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].object == cloud.get());
    // Entry point on the near side of the bound, so the distance is less than
    // the distance to the centre.
    CHECK(hits[0].distance > 0.f);
    CHECK(hits[0].distance < 10.f);
    CHECK(hits[0].point.z > 0.f);
}

TEST_CASE("a ray to one side misses", "[splats]") {

    auto cloud = SplatCloud::create(cloudAt({0.f, 0.f, 0.f}));
    cloud->updateMatrixWorld();

    Raycaster raycaster;
    raycaster.set({100.f, 0.f, 10.f}, {0.f, 0.f, -1.f});

    std::vector<Intersection> hits;
    cloud->raycast(raycaster, hits);

    CHECK(hits.empty());
}

TEST_CASE("a ray pointing away misses", "[splats]") {

    auto cloud = SplatCloud::create(cloudAt({0.f, 0.f, 0.f}));
    cloud->updateMatrixWorld();

    Raycaster raycaster;
    raycaster.set({0.f, 0.f, 10.f}, {0.f, 0.f, 1.f});

    std::vector<Intersection> hits;
    cloud->raycast(raycaster, hits);

    CHECK(hits.empty());
}

TEST_CASE("the bound follows the cloud's transform", "[splats]") {

    // The same cloud moved sideways: the ray that used to hit now misses, and a
    // ray aimed at the new position hits. This is the assertion that the world
    // matrix is applied — a raycast written against the LOCAL sphere passes
    // every test above and fails this one, and an editor gizmo moves things.
    auto cloud = SplatCloud::create(cloudAt({0.f, 0.f, 0.f}));
    cloud->position.set(50.f, 0.f, 0.f);
    cloud->updateMatrixWorld();

    Raycaster raycaster;
    std::vector<Intersection> hits;

    raycaster.set({0.f, 0.f, 10.f}, {0.f, 0.f, -1.f});
    cloud->raycast(raycaster, hits);
    CHECK(hits.empty());

    hits.clear();
    raycaster.set({50.f, 0.f, 10.f}, {0.f, 0.f, -1.f});
    cloud->raycast(raycaster, hits);
    CHECK(hits.size() == 1);
}

TEST_CASE("an invisible or empty cloud is not pickable", "[splats]") {

    SECTION("invisible") {
        auto cloud = SplatCloud::create(cloudAt({0.f, 0.f, 0.f}));
        cloud->visible = false;
        cloud->updateMatrixWorld();

        Raycaster raycaster;
        raycaster.set({0.f, 0.f, 10.f}, {0.f, 0.f, -1.f});

        std::vector<Intersection> hits;
        cloud->raycast(raycaster, hits);
        CHECK(hits.empty());
    }

    SECTION("empty") {
        auto cloud = SplatCloud::create(SplatData{});
        cloud->updateMatrixWorld();

        Raycaster raycaster;
        raycaster.set({0.f, 0.f, 10.f}, {0.f, 0.f, -1.f});

        std::vector<Intersection> hits;
        cloud->raycast(raycaster, hits);
        CHECK(hits.empty());
    }
}
