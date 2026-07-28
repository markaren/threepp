#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadAlignment.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace threepp;

namespace {

    // Straight line, unit tangents: a biarc between them is two straights, and
    // the chain is one piece long once the empty halves fall out.
    BiarcChain::Seed seed(float x, float y, float dx, float dy) {

        const float length = std::sqrt(dx * dx + dy * dy);
        return {Vector2(x, y), Vector2(dx / length, dy / length)};
    }

    void report(const char* what, const RoadAlignment& alignment) {

        const auto& r = alignment.report();
        std::cout << "[" << what << "] seeds=" << r.seeds
                  << " planPieces=" << r.planPieces
                  << " profilePieces=" << r.profilePieces
                  << " stations=" << alignment.stations().size()
                  << " length=" << alignment.length()
                  << "\n           fit=" << r.fit
                  << " deviation=" << r.deviation
                  << " planMinRadius=" << r.planMinRadius
                  << " profileMinRadius=" << r.profileMinRadius
                  << " bendsRelaxed=" << r.bendsRelaxed
                  << " seedsRelaxed=" << r.seedsRelaxed
                  << " planBreak=" << alignment.plan().maxAngleBreak(false)
                  << " profileBreak=" << alignment.profile().maxAngleBreak(false)
                  << std::endl;
    }

    CatmullRomCurve3 defaultSpline() {

        return CatmullRomCurve3({Vector3(-3, 0.5f, 1.5f), Vector3(-1, 0.5f, -1),
                                 Vector3(1, 0.5f, -1), Vector3(3, 0.5f, 1.5f)});
    }

}// namespace


TEST_CASE("a biarc meets both of the tangents it was built from", "[extras]") {

    // The property the whole alignment rests on. Two seeds, tangents 90 degrees
    // apart, and the pair of arcs between them leaves along the first and
    // arrives along the second — exactly, not within a tolerance.
    std::vector<BiarcChain::Seed> seeds{seed(0, 0, 1, 0), seed(4, 4, 0, 1)};
    const auto chain = BiarcChain::fit(seeds, false, BiarcChain::Limits{});

    REQUIRE(chain.pieces().size() == 2);
    const auto& first = chain.pieces().front();
    const auto& second = chain.pieces().back();

    CHECK(std::abs(first.tangent.x - 1.f) < 1e-5f);
    CHECK(std::abs(first.tangent.y) < 1e-5f);
    CHECK(std::abs(second.endTangent().x) < 1e-5f);
    CHECK(std::abs(second.endTangent().y - 1.f) < 1e-5f);
    CHECK(chain.maxAngleBreak(false) < 1e-5f);

    // The joint really is where one arc ends and the next begins.
    CHECK(std::abs(first.end().x - second.start.x) < 1e-5f);
    CHECK(std::abs(first.end().y - second.start.y) < 1e-5f);

    // ...and the far end really is the seed it was aimed at.
    const Vector2 finish = second.end();
    CHECK(std::abs(finish.x - 4.f) < 1e-4f);
    CHECK(std::abs(finish.y - 4.f) < 1e-4f);
}

TEST_CASE("a straight run of seeds makes straights", "[extras]") {

    std::vector<BiarcChain::Seed> seeds{seed(0, 0, 1, 0), seed(3, 0, 1, 0), seed(9, 0, 1, 0)};
    const auto chain = BiarcChain::fit(seeds, false, BiarcChain::Limits{});

    CHECK(std::abs(chain.length() - 9.f) < 1e-4f);
    for (const auto& piece : chain.pieces()) {
        CHECK(piece.curvature == 0.f);
        CHECK(std::isinf(piece.radius()));
    }
    CHECK(chain.minRadius() > 1e9f);
}

TEST_CASE("the minimum-radius clamp relaxes seeds and re-solves", "[extras]") {

    // A right-angle corner: the natural biarc through it is far tighter than
    // three metres, so the clamp has to open it out. What comes back is still a
    // biarc chain — G1 — because the clamp re-solves rather than editing a
    // radius in place.
    std::vector<BiarcChain::Seed> seeds;
    for (int i = 0; i <= 10; ++i) seeds.push_back(seed(-10.f + static_cast<float>(i), 0, 1, 0));
    for (int i = 1; i <= 10; ++i) seeds.push_back(seed(0, static_cast<float>(i), 0, 1));

    BiarcChain::Limits limits;
    limits.minRadius = 3.f;
    BiarcChain::Relaxation relaxation;
    const auto chain = BiarcChain::fit(seeds, false, limits, &relaxation);

    CHECK(chain.minRadius() >= 3.f - 1e-3f);
    CHECK(relaxation.bends > 0);
    CHECK(relaxation.seeds > 0);
    CHECK(chain.maxAngleBreak(false) < 1e-3f);
    // A right angle taken at radius three cuts about 1.24 m off the apex; the
    // relaxation should be spending about that and not several metres.
    CHECK(relaxation.moved < 2.5f);
}

TEST_CASE("an alignment reports what it did to the drawing", "[extras]") {

    auto curve = defaultSpline();

    RoadAlignment::Params narrow;
    narrow.width = 1.f;
    const auto easy = RoadAlignment::build(curve, narrow);
    report("width 1", easy);

    // Nothing to relax at a half-metre half-width, so the alignment IS the
    // curve, to the tolerance it was refined against.
    CHECK(easy.report().bendsRelaxed == 0);
    CHECK(easy.report().fit <= narrow.tolerance);
    CHECK(easy.report().deviation < 0.05f);
    CHECK(easy.plan().maxAngleBreak(false) < 1e-3f);

    RoadAlignment::Params wide;
    wide.width = 4.f;
    const auto relaxed = RoadAlignment::build(curve, wide);
    report("width 4", relaxed);

    // The editor's default spline at the editor's default width: its tightest
    // curvature radius is about 1.35 m and the floor is 2.1, so it cannot be
    // built without bending. That it bends, and by how much, is the whole
    // report.
    CHECK(relaxed.report().planMinRadius >= 2.1f - 1e-3f);
    CHECK(relaxed.report().bendsRelaxed > 0);
    CHECK(relaxed.report().deviation < 1.f);
    CHECK(relaxed.plan().maxAngleBreak(false) < 1e-3f);
    CHECK(relaxed.profile().maxAngleBreak(false) < 1e-3f);
}

TEST_CASE("a profile is a chain of its own over station", "[extras]") {

    // A climb into a crest and down again, with the crest inside a bend.
    CatmullRomCurve3 curve({Vector3(-12, 0, -6), Vector3(-4, 1.5f, -2),
                            Vector3(0, 3, 2), Vector3(4, 1.5f, 6), Vector3(12, 0, 8)});
    RoadAlignment::Params params;
    params.width = 5.f;
    const auto alignment = RoadAlignment::build(curve, params);
    report("crest in turn", alignment);

    CHECK(alignment.profile().maxAngleBreak(false) < 1e-3f);
    CHECK(alignment.report().profileMinRadius >= params.profileMinRadius - 1e-2f);

    // Elevation survives the vertical clamp: the crest is still a crest.
    float highest = -1e9f, lowest = 1e9f;
    for (const auto& station : alignment.stations()) {
        highest = std::max(highest, station.point.y);
        lowest = std::min(lowest, station.point.y);
    }
    CHECK(highest > 2.f);
    CHECK(lowest < 0.5f);

    // The grade turns smoothly: no two consecutive cross-sections differ in
    // pitch by more than the station angle allows.
    const auto& stations = alignment.stations();
    REQUIRE(stations.size() > 8);
    for (std::size_t i = 1; i < stations.size(); ++i) {
        const float before = std::asin(std::clamp(stations[i - 1].tangent.y, -1.f, 1.f));
        const float now = std::asin(std::clamp(stations[i].tangent.y, -1.f, 1.f));
        CHECK(std::abs(now - before) < params.stationAngle * 1.5f);
    }
}

TEST_CASE("stations hold their angle break in plan too", "[extras]") {

    auto curve = defaultSpline();
    RoadAlignment::Params params;
    params.width = 4.f;
    const auto alignment = RoadAlignment::build(curve, params);
    const auto& stations = alignment.stations();
    REQUIRE(stations.size() > 8);

    float worst = 0.f;
    for (std::size_t i = 1; i < stations.size(); ++i) {
        const Vector3& before = stations[i - 1].tangent;
        const Vector3& now = stations[i].tangent;
        worst = std::max(worst, std::acos(std::clamp(before.dot(now), -1.f, 1.f)));
        // Consecutive cross-sections never face away from each other.
        CHECK(stations[i - 1].side.dot(stations[i].side) > 0.f);
        CHECK(stations[i].distance > stations[i - 1].distance);
    }
    std::cout << "[default width 4] worst station break = " << worst << " rad" << std::endl;
    CHECK(worst < params.stationAngle * 1.5f);
}
