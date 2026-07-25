// End-to-end animation: clip -> mixer -> action -> PropertyBinding -> Object3D.
//
// The whole stack was untested. These drive real scene-graph properties through
// AnimationMixer::update and assert the resulting transforms, which is the only
// way to cover PropertyBinding (parses "<node>.<property>" and resolves it
// through a pile of dynamic_casts) and PropertyMixer (accumulates weighted
// contributions from several actions).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/animation/AnimationAction.hpp"
#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/animation/tracks/NumberKeyframeTrack.hpp"
#include "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/objects/Group.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    struct Rig {
        std::shared_ptr<Group> root;
        Object3D* node;
    };

    // A named node under a root, matching how loaders name tracks:
    // "<nodeName>.position".
    Rig makeRig(const std::string& nodeName = "Cube") {
        auto root = Group::create();
        auto child = Object3D::create();
        child->name = nodeName;
        root->add(child);
        return {root, root->children.front()};
    }

    // position track: (0,0,0) at t=0 -> (10,20,30) at t=1
    std::shared_ptr<AnimationClip> makeMoveClip(const std::string& nodeName = "Cube",
                                                float duration = 1.f) {
        std::vector<std::shared_ptr<KeyframeTrack>> tracks{
                std::make_shared<VectorKeyframeTrack>(
                        nodeName + ".position",
                        std::vector<float>{0.f, duration},
                        std::vector<float>{0.f, 0.f, 0.f, 10.f, 20.f, 30.f})};
        return std::make_shared<AnimationClip>("move", duration, tracks);
    }

}// namespace

TEST_CASE("A played clip drives the bound object's position") {

    auto rig = makeRig();
    AnimationMixer mixer(*rig.root);

    auto* action = mixer.clipAction(makeMoveClip());
    REQUIRE(action);
    action->play();

    // Half a second into a 1 s clip = halfway along the track.
    mixer.update(0.5f);
    CHECK_THAT(rig.node->position.x, WithinAbs(5.f, 1e-4));
    CHECK_THAT(rig.node->position.y, WithinAbs(10.f, 1e-4));
    CHECK_THAT(rig.node->position.z, WithinAbs(15.f, 1e-4));

    // ...and the rest of the way.
    mixer.update(0.25f);
    CHECK_THAT(rig.node->position.x, WithinAbs(7.5f, 1e-4));
    CHECK_THAT(rig.node->position.y, WithinAbs(15.f, 1e-4));
}

TEST_CASE("An unplayed action leaves the object alone") {

    auto rig = makeRig();
    rig.node->position.set(1.f, 2.f, 3.f);

    AnimationMixer mixer(*rig.root);
    auto* action = mixer.clipAction(makeMoveClip());
    REQUIRE(action);
    CHECK_FALSE(action->isRunning());

    mixer.update(0.5f);

    CHECK_THAT(rig.node->position.x, WithinAbs(1.f, 1e-5));
    CHECK_THAT(rig.node->position.y, WithinAbs(2.f, 1e-5));
}

TEST_CASE("Loop::Once does not wrap the way Loop::Repeat does") {

    // Differential, because the two modes diverge only after the clip ends:
    // Repeat restarts, Once finishes. (Once does NOT hold the last frame here —
    // clampWhenFinished defaults to false, so the action stops contributing and
    // the property falls back to its unanimated value. That is the documented
    // three.js behaviour, and there is currently no public setter to change it.)
    const float overrun = 1.25f;

    float repeatX = 0.f;
    {
        auto rig = makeRig();
        AnimationMixer mixer(*rig.root);
        auto* action = mixer.clipAction(makeMoveClip());
        action->setLoop(Loop::Repeat, -1);
        action->play();
        mixer.update(overrun);
        repeatX = rig.node->position.x;
    }

    float onceX = 0.f;
    {
        auto rig = makeRig();
        AnimationMixer mixer(*rig.root);
        auto* action = mixer.clipAction(makeMoveClip());
        action->setLoop(Loop::Once, 1);
        action->play();
        mixer.update(overrun);
        onceX = rig.node->position.x;
    }

    INFO("repeat x = " << repeatX << ", once x = " << onceX);
    // Repeat has wrapped a quarter of the way back in.
    CHECK_THAT(repeatX, WithinAbs(2.5f, 1e-3));
    // Once has not restarted the clip.
    CHECK(onceX != repeatX);
}

TEST_CASE("Loop::Repeat wraps back around") {

    auto rig = makeRig();
    AnimationMixer mixer(*rig.root);

    auto* action = mixer.clipAction(makeMoveClip());
    action->setLoop(Loop::Repeat, -1);
    action->play();

    // 1.25 s into a 1 s clip is 0.25 s into the second pass.
    mixer.update(1.25f);
    INFO("x after wrap = " << rig.node->position.x);
    CHECK_THAT(rig.node->position.x, WithinAbs(2.5f, 1e-3));
}

TEST_CASE("Two actions blend by effective weight") {

    // Same node, two clips pulling to different places. At equal weight the
    // result is the average — this is what PropertyMixer accumulates.
    auto rig = makeRig();
    AnimationMixer mixer(*rig.root);

    std::vector<std::shared_ptr<KeyframeTrack>> aTracks{
            std::make_shared<VectorKeyframeTrack>(
                    "Cube.position", std::vector<float>{0.f, 1.f},
                    std::vector<float>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f})};
    std::vector<std::shared_ptr<KeyframeTrack>> bTracks{
            std::make_shared<VectorKeyframeTrack>(
                    "Cube.position", std::vector<float>{0.f, 1.f},
                    std::vector<float>{100.f, 0.f, 0.f, 100.f, 0.f, 0.f})};

    auto* a = mixer.clipAction(std::make_shared<AnimationClip>("a", 1.f, aTracks));
    auto* b = mixer.clipAction(std::make_shared<AnimationClip>("b", 1.f, bTracks));

    a->setEffectiveWeight(0.5f).play();
    b->setEffectiveWeight(0.5f).play();

    mixer.update(0.1f);

    INFO("blended x = " << rig.node->position.x);
    CHECK_THAT(rig.node->position.x, WithinAbs(50.f, 1e-3));
}

TEST_CASE("A quaternion track keeps the rotation normalised") {

    auto rig = makeRig();
    AnimationMixer mixer(*rig.root);

    const float s = std::sqrt(0.5f);
    std::vector<std::shared_ptr<KeyframeTrack>> tracks{
            std::make_shared<QuaternionKeyframeTrack>(
                    "Cube.quaternion",
                    std::vector<float>{0.f, 1.f},
                    std::vector<float>{0.f, 0.f, 0.f, 1.f, 0.f, s, 0.f, s})};

    auto* action = mixer.clipAction(std::make_shared<AnimationClip>("spin", 1.f, tracks));
    action->play();

    for (int i = 0; i < 10; ++i) {
        mixer.update(0.1f);
        const auto& q = rig.node->quaternion;
        const float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        INFO("step " << i << " |q| = " << len);
        REQUIRE_THAT(len, WithinAbs(1.f, 1e-3));
    }
}

TEST_CASE("timeScale scales playback rate") {

    auto rig = makeRig();
    AnimationMixer mixer(*rig.root);

    auto* action = mixer.clipAction(makeMoveClip());
    action->setEffectiveTimeScale(2.f).play();

    // At 2x, 0.25 s of wall time is 0.5 s of clip time.
    mixer.update(0.25f);
    CHECK_THAT(rig.node->position.x, WithinAbs(5.f, 1e-3));
}

TEST_CASE("A track bound to a missing node does not crash the mixer") {

    auto rig = makeRig("Cube");
    AnimationMixer mixer(*rig.root);

    // Name that exists in no node of the graph.
    auto* action = mixer.clipAction(makeMoveClip("NoSuchNode"));
    REQUIRE(action);
    action->play();

    CHECK_NOTHROW(mixer.update(0.5f));

    // The real node must be untouched.
    CHECK_THAT(rig.node->position.x, WithinAbs(0.f, 1e-5));
}

// ---------------------------------------------------------------------------
// AnimationClip
// ---------------------------------------------------------------------------

TEST_CASE("AnimationClip::resetDuration takes the longest track") {

    std::vector<std::shared_ptr<KeyframeTrack>> tracks{
            NumberKeyframeTrack::create("Cube.foo", {0.f, 2.f}, {0.f, 1.f}),
            NumberKeyframeTrack::create("Cube.bar", {0.f, 5.f}, {0.f, 1.f})};

    // Declared duration is deliberately wrong; resetDuration must fix it.
    AnimationClip clip("c", 0.5f, tracks);
    clip.resetDuration();
    CHECK_THAT(clip.getDuration(), WithinAbs(5.f, 1e-5));
}

TEST_CASE("AnimationClip::findByName finds the right clip") {

    std::vector<std::shared_ptr<AnimationClip>> clips{
            std::make_shared<AnimationClip>("walk", 1.f),
            std::make_shared<AnimationClip>("run", 1.f)};

    auto found = AnimationClip::findByName(clips, "run");
    REQUIRE(found);
    CHECK(found->name() == "run");

    CHECK(AnimationClip::findByName(clips, "fly") == nullptr);
}

// ---------------------------------------------------------------------------
// KeyframeTrack transforms
// ---------------------------------------------------------------------------

TEST_CASE("KeyframeTrack shift/scale move the key times") {

    auto track = NumberKeyframeTrack::create("n", {0.f, 1.f, 2.f}, {0.f, 1.f, 2.f});

    track->shift(10.f);
    CHECK_THAT(track->getTimes().front(), WithinAbs(10.f, 1e-5));
    CHECK_THAT(track->getTimes().back(), WithinAbs(12.f, 1e-5));

    track->scale(2.f);
    CHECK_THAT(track->getTimes().front(), WithinAbs(20.f, 1e-5));
    CHECK_THAT(track->getTimes().back(), WithinAbs(24.f, 1e-5));

    // Values are untouched by time transforms.
    CHECK_THAT(track->getValues().back(), WithinAbs(2.f, 1e-5));
}

TEST_CASE("KeyframeTrack reports its value size") {

    auto scalar = NumberKeyframeTrack::create("n", {0.f, 1.f}, {0.f, 1.f});
    CHECK(scalar->getValueSize() == 1);

    auto vector = std::make_shared<VectorKeyframeTrack>(
            "v", std::vector<float>{0.f, 1.f},
            std::vector<float>{0.f, 0.f, 0.f, 1.f, 1.f, 1.f});
    CHECK(vector->getValueSize() == 3);
}

TEST_CASE("KeyframeTrack honours the requested interpolation") {

    const std::vector<float> times{0.f, 1.f};
    const std::vector<float> values{0.f, 0.f, 0.f, 10.f, 10.f, 10.f};

    // NumberKeyframeTrack forwards its interpolation argument.
    auto discreteNumber = NumberKeyframeTrack::create(
            "n", {0.f, 1.f}, {0.f, 10.f}, Interpolation::Discrete);
    CHECK(discreteNumber->getInterpolation() == Interpolation::Discrete);

    // VectorKeyframeTrack must forward it too — its constructor accepts the
    // argument, so silently dropping it means asking for Discrete or Smooth
    // and getting the default with no diagnostic.
    VectorKeyframeTrack discreteVector("v", times, values, Interpolation::Discrete);
    CHECK(discreteVector.getInterpolation() == Interpolation::Discrete);

    VectorKeyframeTrack smoothVector("v", times, values, Interpolation::Smooth);
    CHECK(smoothVector.getInterpolation() == Interpolation::Smooth);

    // Quaternions accept Discrete (glTF STEP: keys are copied verbatim, so the
    // rotation stays normalised) and Linear (slerp), but coerce Smooth, because
    // a cubic evaluates the four components independently and denormalises.
    const std::vector<float> quatValues{0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};

    QuaternionKeyframeTrack stepQuat("q", times, quatValues, Interpolation::Discrete);
    CHECK(stepQuat.getInterpolation() == Interpolation::Discrete);

    QuaternionKeyframeTrack smoothQuat("q", times, quatValues, Interpolation::Smooth);
    CHECK(smoothQuat.getInterpolation() == Interpolation::Linear);

    QuaternionKeyframeTrack defaultQuat("q", times, quatValues);
    CHECK(defaultQuat.getInterpolation() == Interpolation::Linear);
}

TEST_CASE("A STEP rotation track snaps instead of slerping") {

    // glTF's "Step Rotation" case. Two keys 1 s apart: identity, then 90 deg
    // about Y. Discrete must hold the first key for the whole interval and
    // switch at the second, never producing anything in between.
    auto rig = makeRig();
    AnimationMixer mixer(*rig.root);

    const float s = std::sqrt(0.5f);
    std::vector<std::shared_ptr<KeyframeTrack>> tracks{
            std::make_shared<QuaternionKeyframeTrack>(
                    "Cube.quaternion",
                    std::vector<float>{0.f, 1.f},
                    std::vector<float>{0.f, 0.f, 0.f, 1.f, 0.f, s, 0.f, s},
                    Interpolation::Discrete)};

    auto* action = mixer.clipAction(std::make_shared<AnimationClip>("step", 1.f, tracks));
    action->play();

    // Partway through the interval the rotation must still be exactly the
    // first key — a slerp would have it partly rotated by now.
    mixer.update(0.4f);
    INFO("y at t=0.4 = " << rig.node->quaternion.y);
    CHECK_THAT(rig.node->quaternion.y, WithinAbs(0.f, 1e-4));
    CHECK_THAT(rig.node->quaternion.w, WithinAbs(1.f, 1e-4));

    mixer.update(0.4f);
    INFO("y at t=0.8 = " << rig.node->quaternion.y);
    CHECK_THAT(rig.node->quaternion.y, WithinAbs(0.f, 1e-4));
}
