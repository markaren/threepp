// CharacterConfig round trip + the clip matcher.
//
// Same contract as PhysicsConfig/VehicleConfig: the flat key=value string rides
// through userData, unknown keys survive a newer-editor document, and PRESENCE
// (not an enabled= key) is what makes a node a character. The per-gait clip
// names live on plain keys of their own because a clip name may contain the
// flat format's delimiters.
//
// The matcher cases are the feature's soul: which clip is the walk, which is
// the run, which is a strafe and which way it goes are read off each clip's own
// ROOT MOTION rather than off its name — so a locomotion pack works with
// nothing typed, whatever its author called the files. PhysX-free, so it runs
// everywhere.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/CharacterConfig.hpp"

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    // One clip whose root bone travels (dx, dz) metres over `duration`. `ramp`
    // makes it accelerate from a standstill instead of holding one speed —
    // which is what a "start walking" transition does, and what the matcher has
    // to throw out.
    std::shared_ptr<AnimationClip> makeClip(const std::string& name, float duration,
                                            float dx, float dz, bool ramp = false) {

        std::vector<float> times;
        std::vector<float> values;
        const int steps = ramp ? 8 : 1;
        for (int i = 0; i <= steps; ++i) {
            const float u = static_cast<float>(i) / static_cast<float>(steps);
            const float travelled = ramp ? u * u * u : u;
            times.push_back(u * duration);
            values.push_back(dx * travelled);
            values.push_back(0.9f);
            values.push_back(dz * travelled);
        }
        std::vector<std::shared_ptr<KeyframeTrack>> tracks{
                std::make_shared<VectorKeyframeTrack>("Hips.position", times, values)};
        return std::make_shared<AnimationClip>(name, duration, tracks);
    }

    // A rig the derivation can measure: a 1.8 m box body 0.3 m deep over one
    // "Hips" bone, carrying a full locomotion set whose speeds are known.
    // Deliberately named nothing like the roles they fill, so a matcher that
    // cheated by reading names would fail every case below.
    std::shared_ptr<Group> makeRig() {

        auto rig = Group::create();
        rig->name = "Hero";

        auto body = Mesh::create(BoxGeometry::create(0.4f, 1.8f, 0.3f));
        body->name = "Body";
        body->position.set(0.f, 0.9f, 0.f);
        rig->add(body);

        auto hips = Bone::create();
        hips->name = "Hips";
        hips->position.set(0.f, 0.9f, 0.f);
        rig->add(hips);

        rig->animations = {
                makeClip("A_idle", 4.f, 0.f, 0.f),
                makeClip("B", 1.f, 0.f, 1.4f),      // forward, slow  -> Walk
                makeClip("C", 0.5f, 0.f, 2.f),      // forward, fast  -> Run
                makeClip("D", 1.f, 0.f, -1.1f),     // backward, slow -> WalkBack
                makeClip("E", 0.5f, 0.f, -1.3f),    // backward, fast -> RunBack
                makeClip("F", 1.f, 1.2f, 0.f),      // left, slow
                makeClip("G", 0.5f, 1.6f, 0.f),     // left, fast
                makeClip("H", 1.f, -1.2f, 0.f),     // right, slow
                makeClip("I", 0.5f, -1.6f, 0.f),    // right, fast
                makeClip("J_jump", 1.f, 0.f, 0.f),
                makeClip("K", 2.f, 0.f, 2.4f, true),// a transition, not a cycle
        };
        return rig;
    }

    std::string nameOf(const CharacterGeometry& geo, Gait gait) {

        const auto& slot = geo.slot(gait);
        return slot.clip ? slot.clip->name() : std::string();
    }

}// namespace


TEST_CASE("CharacterConfig round trips through userData", "[editor][character]") {

    CharacterConfig config;
    config.facing = CharacterConfig::Facing::Movement;
    config.autoGeometry = false;
    config.height = 1.62f;
    config.radius = 0.27f;
    config.autoSpeeds = false;
    config.walkSpeed = 1.15f;
    config.runSpeed = 5.5f;
    config.mass = 82.f;
    config.jumpHeight = 1.25f;
    config.gravity = 22.f;
    config.stepOffset = 0.4f;
    config.slopeLimit = math::degToRad(38.f);
    config.turnRate = 9.f;
    config.accel = 11.f;
    config.blendTime = 0.22f;

    const auto decoded = CharacterConfig::decode(config.encode());
    REQUIRE(decoded);

    // Field by field with a tolerance, not operator==, and deliberately: the
    // codec is fixed 6-decimal text (see ConfigCodec), so a value that is not
    // representable in six decimals — an angle typed in degrees and stored in
    // radians is the everyday case — comes back a few ulps off. That is the
    // format's documented contract, not a bug to assert away.
    CHECK(decoded->facing == CharacterConfig::Facing::Movement);
    CHECK_FALSE(decoded->autoGeometry);
    CHECK_THAT(decoded->height, WithinAbs(1.62f, 1e-4f));
    CHECK_THAT(decoded->radius, WithinAbs(0.27f, 1e-4f));
    CHECK_FALSE(decoded->autoSpeeds);
    CHECK_THAT(decoded->walkSpeed, WithinAbs(1.15f, 1e-4f));
    CHECK_THAT(decoded->runSpeed, WithinAbs(5.5f, 1e-4f));
    CHECK_THAT(decoded->mass, WithinAbs(82.f, 1e-4f));
    CHECK_THAT(decoded->jumpHeight, WithinAbs(1.25f, 1e-4f));
    CHECK_THAT(decoded->gravity, WithinAbs(22.f, 1e-4f));
    CHECK_THAT(decoded->stepOffset, WithinAbs(0.4f, 1e-4f));
    CHECK_THAT(decoded->slopeLimit, WithinAbs(math::degToRad(38.f), 1e-4f));
    CHECK_THAT(decoded->turnRate, WithinAbs(9.f, 1e-4f));
    CHECK_THAT(decoded->accel, WithinAbs(11.f, 1e-4f));
    CHECK_THAT(decoded->blendTime, WithinAbs(0.22f, 1e-4f));

    // And a value that IS representable round trips byte-identically, which is
    // what keeps a saved document diff-clean.
    config.slopeLimit = 0.875f;
    CHECK(config.encode() == CharacterConfig::decode(config.encode())->encode());
}

TEST_CASE("presence is what makes a node a character", "[editor][character]") {

    auto node = Group::create();
    CHECK_FALSE(CharacterConfig::isCharacter(*node));
    CHECK_FALSE(CharacterConfig::read(*node).has_value());

    // Defaults included: unlike PhysicsConfig, a default character still
    // writes, because the entry IS the authoring.
    CharacterConfig{}.write(*node);
    CHECK(CharacterConfig::isCharacter(*node));
    REQUIRE(CharacterConfig::read(*node).has_value());

    CharacterConfig::erase(*node);
    CHECK_FALSE(CharacterConfig::isCharacter(*node));
}

TEST_CASE("clip names ride their own keys, delimiters and all", "[editor][character]") {

    auto node = Group::create();
    CharacterConfig config;
    // A name containing both of the flat format's delimiters. On the shared
    // string it would split the entry in two; on its own key it survives.
    config.clips[gaitIndex(Gait::Walk)] = "walk;fast=no";
    config.write(*node);

    const auto read = CharacterConfig::read(*node);
    REQUIRE(read);
    CHECK(read->clips[gaitIndex(Gait::Walk)] == "walk;fast=no");
    // And the flat entry is still parsable around it.
    CHECK(read->facing == config.facing);
    CHECK_FALSE(read->clips[gaitIndex(Gait::Run)].size());
}

TEST_CASE("an unknown key survives a newer-editor document", "[editor][character]") {

    auto node = Group::create();
    node->userData[CharacterConfig::userDataKey] =
            std::string("facing=movement;height=1.5;somethingNew=42");

    const auto read = CharacterConfig::read(*node);
    REQUIRE(read);
    CHECK(read->facing == CharacterConfig::Facing::Movement);
    CHECK_THAT(read->height, WithinAbs(1.5f, 1e-4f));
}

TEST_CASE("the capsule is measured off the model", "[editor][character]") {

    auto rig = makeRig();
    const auto geo = CharacterConfig{}.derived(*rig);

    REQUIRE(geo.valid);
    CHECK_THAT(geo.height, WithinAbs(1.8f, 0.02f));
    // Half-DEPTH floored at human proportion (0.17 * height = 0.306), never
    // half-width: a rigged character binds in a T-pose, so its X extent is an
    // arm span.
    CHECK_THAT(geo.radius, WithinAbs(0.306f, 0.02f));
    REQUIRE(geo.rootBone != nullptr);
    CHECK(geo.rootBone->name == "Hips");
}

TEST_CASE("each clip's role is read off its root motion, not its name",
          "[editor][character]") {

    auto rig = makeRig();
    const auto geo = CharacterConfig{}.derived(*rig);
    REQUIRE(geo.valid);

    // Forward: slowest is the walk, fastest is the run.
    CHECK(nameOf(geo, Gait::Walk) == "B");
    CHECK(nameOf(geo, Gait::Run) == "C");
    CHECK_THAT(geo.slot(Gait::Walk).speed, WithinAbs(1.4f, 0.02f));
    CHECK_THAT(geo.slot(Gait::Run).speed, WithinAbs(4.f, 0.02f));

    CHECK(nameOf(geo, Gait::WalkBack) == "D");
    CHECK(nameOf(geo, Gait::RunBack) == "E");
    // +X is the model's LEFT in the three.js frame this editor uses throughout.
    CHECK(nameOf(geo, Gait::StrafeLeft) == "F");
    CHECK(nameOf(geo, Gait::StrafeLeftFast) == "G");
    CHECK(nameOf(geo, Gait::StrafeRight) == "H");
    CHECK(nameOf(geo, Gait::StrafeRightFast) == "I");

    // Idle and jump ARE by name: standing still and jumping on the spot look
    // identical to a ruler.
    CHECK(nameOf(geo, Gait::Idle) == "A_idle");
    CHECK(nameOf(geo, Gait::Jump) == "J_jump");

    CHECK(geo.resolvedCount() == kGaitCount);
}

TEST_CASE("a ramping transition clip is never adopted as a gait",
          "[editor][character]") {

    auto rig = makeRig();
    const auto geo = CharacterConfig{}.derived(*rig);
    REQUIRE(geo.valid);

    // "K" travels 2.4 m in 2 s — an average of 1.2 m/s, which sits right
    // between the walk and the backward clips and would otherwise win the
    // slow-forward slot. It is rejected because its speed RAMPS: a controller
    // holding 1.2 m/s would foot-slide through the whole clip.
    for (std::size_t i = 0; i < kGaitCount; ++i) {
        const auto& slot = geo.gaits[i];
        CHECK((!slot.clip || slot.clip->name() != "K"));
    }
}

TEST_CASE("an explicit clip name overrides the match", "[editor][character]") {

    auto rig = makeRig();
    CharacterConfig config;
    config.clips[gaitIndex(Gait::Walk)] = "C";// the run clip, on purpose

    const auto geo = config.derived(*rig);
    REQUIRE(geo.valid);
    CHECK(nameOf(geo, Gait::Walk) == "C");
    // And it is re-measured, so the anti-foot-slide speed follows the override
    // rather than the slot it landed in.
    CHECK_THAT(geo.slot(Gait::Walk).speed, WithinAbs(4.f, 0.02f));
}

TEST_CASE("a named clip that is not in the model is reported, not guessed at",
          "[editor][character]") {

    auto rig = makeRig();
    CharacterConfig config;
    config.clips[gaitIndex(Gait::Run)] = "no such clip";

    const auto geo = config.derived(*rig);
    REQUIRE(geo.valid);
    CHECK(geo.problem.find("no such clip") != std::string::npos);
    // The auto-match's answer is left standing rather than being cleared: a
    // typo must not silently stop the character from running.
    CHECK(nameOf(geo, Gait::Run) == "C");
}

TEST_CASE("a model with no clips still derives a capsule", "[editor][character]") {

    auto rig = makeRig();
    rig->animations.clear();

    const auto geo = CharacterConfig{}.derived(*rig);
    // Valid, because a capsule with no clips still walks around — a better
    // first Play than refusing to start.
    CHECK(geo.valid);
    CHECK(geo.resolvedCount() == 0);
    CHECK_FALSE(geo.problem.empty());
    CHECK_THAT(geo.height, WithinAbs(1.8f, 0.02f));
}

TEST_CASE("a rotated character measures and classifies in its own frame",
          "[editor][character]") {

    // Turned a quarter turn in the scene: "forward" is the model's +Z, not the
    // world's, so every role must come out exactly as before.
    auto rig = makeRig();
    rig->rotation.y = math::PI * 0.5f;
    rig->position.set(7.f, 0.f, -3.f);

    const auto geo = CharacterConfig{}.derived(*rig);
    REQUIRE(geo.valid);
    CHECK_THAT(geo.height, WithinAbs(1.8f, 0.02f));
    CHECK(nameOf(geo, Gait::Walk) == "B");
    CHECK(nameOf(geo, Gait::WalkBack) == "D");
    CHECK(nameOf(geo, Gait::StrafeLeft) == "F");
    CHECK(nameOf(geo, Gait::StrafeRight) == "H");
    // And the feet are where the model is standing, not at the origin.
    CHECK_THAT(geo.feet.x, WithinAbs(7.f, 0.02f));
    CHECK_THAT(geo.feet.z, WithinAbs(-3.f, 0.02f));
    CHECK_THAT(geo.feet.y, WithinAbs(0.f, 0.02f));
}
