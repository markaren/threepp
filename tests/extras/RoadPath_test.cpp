#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadPath.hpp"

#include <cmath>
#include <vector>

using namespace threepp;

namespace {

    constexpr float kPi = 3.14159265358979f;

    // Every piece has to start where the one before it ended, or the road built
    // from them has a crack in it.
    void checkChained(const RoadPath& path) {

        const auto& primitives = path.primitives();
        for (std::size_t i = 0; i + 1 < primitives.size(); ++i) {
            CHECK(primitives[i].end.distanceTo(primitives[i + 1].start) < 1e-4f);
        }
        if (path.closed() && !primitives.empty()) {
            CHECK(primitives.back().end.distanceTo(primitives.front().start) < 1e-4f);
        }
    }

    // Samples of an arc around `center`, angles in radians.
    std::vector<Vector3> arcSamples(const Vector3& center, float radius, float from, float to,
                                    int steps, float y = 0.f) {

        std::vector<Vector3> out;
        out.reserve(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            const float angle = from + (to - from) * static_cast<float>(i) / static_cast<float>(steps);
            out.emplace_back(center.x + radius * std::cos(angle), y, center.z + radius * std::sin(angle));
        }
        return out;
    }

}// namespace


TEST_CASE("a straight polyline is one straight", "[extras]") {

    std::vector<Vector3> samples;
    for (int i = 0; i <= 20; ++i) samples.emplace_back(static_cast<float>(i) * 0.5f, 0.f, 3.f);

    const auto path = RoadPath::fromPoints(samples, false);

    REQUIRE(path.primitives().size() == 1);
    const auto& straight = path.primitives().front();
    CHECK(straight.kind == RoadPrimitive::Kind::Straight);
    CHECK(straight.start.distanceTo(Vector3(0, 0, 3)) < 1e-5f);
    CHECK(straight.end.distanceTo(Vector3(10, 0, 3)) < 1e-5f);
    CHECK(std::abs(path.length() - 10.f) < 1e-4f);
}

TEST_CASE("a circle sampling is one arc", "[extras]") {

    // Three quarters of a circle, nothing else: the fit has to come back with
    // the circle it was sampled from — centre, radius and signed sweep — rather
    // than a chain of chords.
    const Vector3 center(2.f, 0.f, -1.f);
    const auto samples = arcSamples(center, 5.f, 0.f, 1.5f * kPi, 90, 0.25f);

    const auto path = RoadPath::fromPoints(samples, false);

    REQUIRE(path.primitives().size() == 1);
    const auto& arc = path.primitives().front();
    CHECK(arc.kind == RoadPrimitive::Kind::Arc);
    CHECK(std::abs(arc.center.x - center.x) < 1e-3f);
    CHECK(std::abs(arc.center.z - center.z) < 1e-3f);
    CHECK(std::abs(arc.radius - 5.f) < 1e-3f);
    CHECK(std::abs(arc.startAngle) < 1e-3f);
    CHECK(std::abs(arc.sweep - 1.5f * kPi) < 1e-3f);
    // Arc length, not chord length.
    CHECK(std::abs(path.length() - 5.f * 1.5f * kPi) < 1e-2f);
    CHECK(arc.pointAt(0.f).distanceTo(samples.front()) < 1e-4f);
    CHECK(arc.pointAt(1.f).distanceTo(samples.back()) < 1e-4f);
}

TEST_CASE("an L corner is straight, arc, straight", "[extras]") {

    // A road that runs along +X, turns a filleted right angle and leaves along
    // +Z. The two legs are exactly straight and the corner exactly circular, so
    // the segmentation has no excuse for anything else.
    std::vector<Vector3> samples;
    for (int i = 0; i <= 32; ++i) samples.emplace_back(-10.f + static_cast<float>(i) * 0.25f, 0.f, 0.f);
    for (auto& point : arcSamples(Vector3(-2.f, 0.f, 2.f), 2.f, -0.5f * kPi, 0.f, 60)) {
        if (point.distanceTo(samples.back()) > 1e-5f) samples.push_back(point);
    }
    for (int i = 1; i <= 32; ++i) samples.emplace_back(0.f, 0.f, 2.f + static_cast<float>(i) * 0.25f);

    const auto path = RoadPath::fromPoints(samples, false);
    checkChained(path);

    REQUIRE(path.primitives().size() == 3);
    CHECK(path.primitives()[0].kind == RoadPrimitive::Kind::Straight);
    CHECK(path.primitives()[1].kind == RoadPrimitive::Kind::Arc);
    CHECK(path.primitives()[2].kind == RoadPrimitive::Kind::Straight);

    const auto& arc = path.primitives()[1];
    CHECK(std::abs(arc.center.x + 2.f) < 1e-2f);
    CHECK(std::abs(arc.center.z - 2.f) < 1e-2f);
    CHECK(std::abs(arc.radius - 2.f) < 1e-2f);
    // A greedy straight runs as far as its own tolerance allows, so it takes
    // the first few degrees of the corner with it: the arc turns a little less
    // than the quarter it was sampled as, and in the same direction.
    CHECK(arc.sweep > 0.f);
    CHECK(arc.sweep < 0.5f * kPi + 1e-3f);
    CHECK(arc.sweep > 0.5f * kPi - 0.35f);
}

TEST_CASE("the editor's default spline is a handful of pieces", "[extras]") {

    // ObjectFactory::createSpline's arc at its default tessellation — 24
    // samples per segment over three segments. The point of the consolidation
    // is that 72 spans come out as a road that can be described in one line.
    CatmullRomCurve3 curve({Vector3(-3, 0.5f, 1.5f), Vector3(-1, 0.5f, -1),
                            Vector3(1, 0.5f, -1), Vector3(3, 0.5f, 1.5f)});

    const auto path = RoadPath::fromCurve(curve, 72, false);
    checkChained(path);

    CHECK(!path.primitives().empty());
    CHECK(path.primitives().size() <= 8);
    // One bend in the middle, so at least one piece is a real arc.
    bool hasArc = false;
    for (const auto& primitive : path.primitives()) {
        if (primitive.kind == RoadPrimitive::Kind::Arc) hasArc = true;
    }
    CHECK(hasArc);

    // The chain is the curve, not a shortcut across it: chords would come up
    // short, and a fit that wandered would come up long.
    const float length = curve.getLength();
    CHECK(path.length() > length * 0.99f);
    CHECK(path.length() < length * 1.01f);

    // ...and it starts and ends where the curve does.
    Vector3 first, last;
    curve.getPoint(0.f, first);
    curve.getPoint(1.f, last);
    CHECK(path.primitives().front().start.distanceTo(first) < 1e-4f);
    CHECK(path.primitives().back().end.distanceTo(last) < 1e-4f);
}

TEST_CASE("a closed spline closes its chain", "[extras]") {

    CatmullRomCurve3 curve({Vector3(-4, 0, -4), Vector3(4, 0, -4),
                            Vector3(4, 0, 4), Vector3(-4, 0, 4)},
                           /*closed*/ true);

    const auto path = RoadPath::fromCurve(curve, 96, true);

    CHECK(path.closed());
    CHECK(path.primitives().size() >= 4);// four corners, at least
    checkChained(path);
}

TEST_CASE("a hill splits into pieces that keep its grade", "[extras]") {

    // Straight in XZ, but climbing over a crest. The height profile of a piece
    // is LINEAR, so a road that is one straight in plan has to break into
    // several to follow it — flattening the hill into one slab is the failure
    // this pins.
    std::vector<Vector3> samples;
    for (int i = 0; i <= 60; ++i) {
        const float z = static_cast<float>(i) * 0.5f;
        samples.emplace_back(0.f, 3.f * std::sin(z * 0.1f), z);
    }

    const auto path = RoadPath::fromPoints(samples, false);
    checkChained(path);

    CHECK(path.primitives().size() > 1);
    for (const auto& primitive : path.primitives()) {
        CHECK(primitive.kind == RoadPrimitive::Kind::Straight);
    }

    // Every sample is within the fit tolerance of the chained profile.
    float worst = 0.f;
    for (const auto& sample : samples) {
        float best = 1e30f;
        for (const auto& primitive : path.primitives()) {
            const float span = primitive.end.z - primitive.start.z;
            if (sample.z < primitive.start.z - 1e-4f || sample.z > primitive.end.z + 1e-4f) continue;
            const float t = std::abs(span) < 1e-6f ? 0.f : (sample.z - primitive.start.z) / span;
            best = std::min(best, std::abs(primitive.pointAt(t).y - sample.y));
        }
        worst = std::max(worst, best);
    }
    CHECK(worst < 0.06f);
}
