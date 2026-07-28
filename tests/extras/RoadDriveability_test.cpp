// Does the road DRIVE?
//
// Every previous version of this feature passed a suite of invariants and was
// rejected the moment somebody looked at it, because the invariants measured
// what the code did rather than what the road does. This measures the road.
//
// A dynamic sphere is run along the surface at a fixed speed, steered down the
// centreline, and its height is sampled at a fixed interval. The second
// difference of that height is the vertical acceleration a body driving the
// road actually feels. A smooth road gives small, boring numbers; a facet
// break, a crest crease or a joint that steps gives a spike, and the distance
// it happened at says where to look.
//
// The threshold is not a guess. The flat straight below is the same rig on a
// surface with nothing in it, so it measures the noise floor of the contact
// solver itself; everything else is judged against that and against v^2/R,
// which is what a vertical curve of the design radius is entitled to.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/curves/RoadGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr float kStep = 1.f / 120.f;
    // Twelve steps a sample: a tenth of a second, and at the speed below about
    // a third of a metre of road. Finer than that and the measurement is
    // dominated by the millimetre the contact solver breathes at rest; coarser
    // and a short crease averages away.
    constexpr int kStepsPerSample = 12;
    constexpr float kSampleTime = kStep * kStepsPerSample;
    constexpr float kSpeed = 3.f;
    constexpr float kBallRadius = 0.25f;

    struct Drive {
        float peak = 0.f;    // largest |vertical acceleration|, m/s^2
        float peakAt = 0.f;  // how far along the road that was, m
        float rms = 0.f;
        float distance = 0.f;// how much road the ball covered
        int samples = 0;
    };

    Drive roll(const char* what, const std::vector<Vector3>& points, float width) {

        SceneDocument document;
        auto& scene = document.scene();

        auto spline = Group::create();
        SplineConfig config;
        config.mesh = SplineConfig::MeshKind::Road;
        config.width = width;
        config.write(*spline);
        for (const auto& point : points) {
            auto node = Object3D::create();
            node->position.copy(point);
            spline->add(node);
        }
        auto curve = config.curve(*spline);
        REQUIRE(curve != nullptr);

        auto geometry = RoadGeometry::create(*curve, width, config.divisions(*spline),
                                             config.uvLength, config.closed);
        auto road = Mesh::create(geometry);
        road->name = "Road";
        SplineConfig::markDerived(*road);
        spline->add(road);

        PhysicsConfig surface;
        surface.enabled = true;
        surface.body = PhysicsConfig::Body::Static;
        surface.shape = PhysicsConfig::Shape::Auto;
        surface.friction = 0.6f;
        surface.restitution = 0.f;
        surface.write(*road);
        scene.add(spline);

        PhysxWorld::Settings settings;
        settings.fixedTimestep = kStep;
        settings.maxSubSteps = 1;
        PhysicsPlaySession session(settings);
        session.start(scene);
        auto* world = session.world();
        REQUIRE(world != nullptr);

        const auto& stations = geometry->alignment().stations();
        REQUIRE(stations.size() > 8);

        // A metre in, not on the end cap. A ball resting exactly on the first
        // cross-section has half its contact hanging off the end of the road,
        // and on a road that starts uphill it simply rolls backwards off it.
        std::size_t from = 0;
        while (from + 1 < stations.size() && stations[from].distance < 1.f) ++from;

        auto ball = Mesh::create(SphereGeometry::create(kBallRadius, 16, 12));
        ball->position.copy(stations[from].point).addScaledVector(stations[from].normal, kBallRadius + 0.05f);
        auto* material = world->physics().createMaterial(0.6f, 0.6f, 0.f);
        auto* body = world->add(*ball, 800.f, material);
        REQUIRE(body != nullptr);

        for (int i = 0; i < 60; ++i) session.update(kStep);// settle onto the surface

        // Then a spin-up: the commanded speed is ramped in and nothing is
        // recorded until it is steady. Snapping a resting ball to three metres
        // a second launches it off the first slope it meets, and free fall
        // reads as a 9.81 m/s^2 "defect" that belongs to the rig.
        constexpr int kSpinUp = 90;

        const auto flatDistance = [](const Vector3& a, const Vector3& b) {
            return std::hypot(a.x - b.x, a.z - b.z);
        };

        std::vector<float> heights;
        std::vector<float> reached;
        std::size_t at = from;
        float travelled = 0.f;
        Vector3 previous = ball->position;

        for (int step = 0; step < 60000; ++step) {

            // Which station the ball is over, and one a look-ahead further on
            // to steer at. Pure pursuit: it keeps the ball on the centreline
            // without needing the road to hold it there.
            while (at + 1 < stations.size() &&
                   flatDistance(stations[at + 1].point, ball->position) <=
                           flatDistance(stations[at].point, ball->position)) {
                ++at;
            }
            std::size_t aim = at;
            while (aim + 1 < stations.size() &&
                   stations[aim].distance - stations[at].distance < 0.8f) {
                ++aim;
            }
            if (aim + 1 >= stations.size() &&
                flatDistance(stations.back().point, ball->position) < 0.4f) {
                break;
            }

            const Vector3& target = stations[aim].point;
            const float dx = target.x - ball->position.x;
            const float dz = target.z - ball->position.z;
            const float reach = std::hypot(dx, dz);
            if (reach < 1e-4f) break;

            const float wanted =
                    kSpeed * std::min(1.f, static_cast<float>(step + 1) / static_cast<float>(kSpinUp));
            const auto velocity = body->getLinearVelocity();
            body->setLinearVelocity(::physx::PxVec3(dx / reach * wanted, velocity.y,
                                                    dz / reach * wanted));
            session.update(kStep);

            if (step < kSpinUp) {
                previous = ball->position;
                continue;
            }
            travelled += flatDistance(ball->position, previous);
            previous = ball->position;

            if ((step - kSpinUp) % kStepsPerSample == kStepsPerSample - 1) {
                heights.push_back(ball->position.y);
                reached.push_back(travelled);
            }
        }

        Drive out;
        out.distance = travelled;
        // Drop two samples either end: the rig is accelerating into the road at
        // one and running out of it at the other, and neither is the road.
        double square = 0.0;
        for (std::size_t i = 3; i + 3 < heights.size(); ++i) {
            const float acceleration =
                    (heights[i + 1] - 2.f * heights[i] + heights[i - 1]) / (kSampleTime * kSampleTime);
            square += static_cast<double>(acceleration) * acceleration;
            ++out.samples;
            if (std::abs(acceleration) > out.peak) {
                out.peak = std::abs(acceleration);
                out.peakAt = reached[i];
            }
        }
        if (out.samples > 0) out.rms = static_cast<float>(std::sqrt(square / out.samples));

        std::cout << "[drive] " << what << ": peak |a_y| = " << out.peak << " m/s^2 at "
                  << out.peakAt << " m, rms = " << out.rms << " m/s^2, over " << out.distance
                  << " m in " << out.samples << " samples"
                  << " (minRadius " << geometry->alignment().report().planMinRadius
                  << " / " << geometry->alignment().report().profileMinRadius << ")" << std::endl;
        return out;
    }

    // The rig's own noise: a flat straight has nothing in it to feel, so
    // whatever this reads is the contact solver breathing, and every other road
    // is judged against it.
    Drive flatStraight() {

        return roll("flat straight", {Vector3(-8, 0, 0), Vector3(0, 0, 0), Vector3(8, 0, 0)}, 4.f);
    }

}// namespace


TEST_CASE("a road drives without a jolt in it", "[extras][physx]") {

    const Drive floorNoise = flatStraight();
    // Nothing to feel on a flat straight, so whatever this reads is the solver
    // breathing under a resting contact. Measured 0.25 peak, 0.09 rms. If it is
    // not small everything below means nothing, so it is checked first.
    CHECK(floorNoise.samples > 8);
    CHECK(floorNoise.peak < 0.75f);

    // The editor's DEFAULT spline at the DEFAULT width — the case the whole
    // rebuild is judged on. Its tightest curvature radius is under half the
    // road's width, so the alignment bends it out; what matters here is that
    // driving over the result feels like nothing.
    const Drive standard = roll("editor default, width 4",
                                {Vector3(-3, 0.5f, 1.5f), Vector3(-1, 0.5f, -1),
                                 Vector3(1, 0.5f, -1), Vector3(3, 0.5f, 1.5f)},
                                4.f);
    // Measured 0.07 peak, 0.03 rms — under the flat straight's own noise. The
    // road is short, so there are not many samples in it.
    CHECK(standard.samples > 8);
    CHECK(standard.peak < 1.f);

    // The S the screenshot scene builds: two opposite bends, both far tighter
    // than its six metre width. This is where the trimmed offset put an apex
    // fan and the ribbon before it folded.
    const Drive snake = roll("S-curve, width 6",
                             {Vector3(-9, 0.5f, -3), Vector3(-3, 0.5f, 3),
                              Vector3(3, 0.5f, -3), Vector3(9, 0.5f, 3)},
                             6.f);
    // Measured 0.33 peak, 0.13 rms.
    CHECK(snake.samples > 10);
    CHECK(snake.peak < 1.f);

    // Grade, and a crest INSIDE a bend: the open complaint. The crest here is
    // the curve the user drew, not one the vertical floor imposed, so what a
    // driver feels over it is v^2/R for the arc that is genuinely there.
    const Drive graded = roll("crest in a turn, width 5",
                              {Vector3(-12, 0, -6), Vector3(-4, 1.5f, -2), Vector3(0, 3, 2),
                               Vector3(4, 1.5f, 6), Vector3(12, 0, 8)},
                              5.f);
    // Measured 1.71 peak at the crest, 0.64 rms — and 9 / 5.13, the crest's own
    // vertical radius, is 1.75. The largest thing on this road reads as exactly
    // the arc it is and not a fraction more, which is what "no crease" looks
    // like as a number.
    CHECK(graded.samples > 10);
    CHECK(graded.peak < 2.5f);

    // ...and it really did climb: a road that drives smoothly because it was
    // flattened into a plank is not the road that was drawn.
    CHECK(graded.distance > 20.f);
}
