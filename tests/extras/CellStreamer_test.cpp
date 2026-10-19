
#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/terrain/CellStreamer.hpp"
#include "threepp/objects/Group.hpp"

using namespace threepp;
using namespace threepp::terrain;

TEST_CASE("CellStreamer follows the camera and caches what it built") {

    int builds = 0;
    CellStreamerOptions o;
    o.fineCellSize = 100.f;
    o.fineRadius = 150.f;
    o.coarseRadius = 0.f;// fine only, so the test is about the ring
    o.removeSlack = 1.3f;
    o.maxCellBuildsPerFrame = 1000;
    o.evictRadius = 0.f;// keep every built cell: the come-back case below

    auto s = CellStreamer::create(
            [&](int, int, int, float) -> std::shared_ptr<Object3D> {
                ++builds;
                return Group::create();
            },
            o);

    s->update(Vector3(0, 0, 0));
    const int atOrigin = s->stats().active;
    REQUIRE(atOrigin > 0);
    REQUIRE(builds == atOrigin);

    // Fly 2 km: the origin cells go, a whole new set arrives — the defect this
    // class exists for (content used to live in one box around the startup
    // camera and never move).
    s->update(Vector3(2000, 0, 0));
    REQUIRE(s->stats().removes == atOrigin);
    REQUIRE(s->stats().builds > 0);
    const int away = static_cast<int>(builds);

    // Come back: same cells, re-attached from the cache, nothing rebuilt.
    s->update(Vector3(0, 0, 0));
    REQUIRE(builds == away);
    REQUIRE(s->stats().active == atOrigin);

    // Hysteresis: a step that leaves the radius but stays inside radius ×
    // removeSlack must not drop a single cell.
    const int before = s->stats().active;
    s->update(Vector3(30, 0, 0));
    REQUIRE(s->stats().removes == 0);
    REQUIRE(s->stats().active >= before);
}

TEST_CASE("CellStreamer drops cached cells beyond the outer radius x removeSlack by default") {

    int builds = 0;
    CellStreamerOptions o;
    o.fineCellSize = 100.f;
    o.fineRadius = 150.f;
    o.coarseRadius = 0.f;
    o.removeSlack = 1.3f;
    o.maxCellBuildsPerFrame = 1000;
    REQUIRE(o.evictRadius < 0.f);// the default: the outer radius x removeSlack

    auto s = CellStreamer::create(
            [&](int, int, int, float) -> std::shared_ptr<Object3D> {
                ++builds;
                return Group::create();
            },
            o);

    s->update(Vector3(0, 0, 0));
    const int atOrigin = s->stats().active;
    REQUIRE(atOrigin > 0);

    // 2 km away: the origin cells leave the ring and the cache with it (their
    // GPU assets are freed), so memory does not grow with the ground covered.
    s->update(Vector3(2000, 0, 0));
    REQUIRE(s->stats().removes == atOrigin);
    REQUIRE(s->stats().evictions >= atOrigin);
    REQUIRE(s->stats().cached == s->stats().active);

    // Coming back builds them again.
    const int before = builds;
    s->update(Vector3(0, 0, 0));
    REQUIRE(builds == before + atOrigin);
    REQUIRE(s->stats().active == atOrigin);

    // A step inside radius x removeSlack keeps every cell, attached or cached.
    s->update(Vector3(30, 0, 0));
    REQUIRE(s->stats().removes == 0);
    REQUIRE(s->stats().evictions == 0);
}
