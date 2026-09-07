
#include <catch2/catch_test_macros.hpp>

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"
#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/AnimationPlaySession.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A group carrying one clip that raises the named node from y=0 to y=2
    // over one second.
    std::shared_ptr<Group> animatedGroup(const std::string& name) {

        auto group = Group::create();
        group->name = name;

        auto track = std::make_shared<VectorKeyframeTrack>(
                name + ".position",
                std::vector<float>{0.f, 1.f},
                std::vector<float>{0.f, 0.f, 0.f, 0.f, 2.f, 0.f});
        group->animations.push_back(std::make_shared<AnimationClip>(
                "Rise", 1.f, std::vector<std::shared_ptr<KeyframeTrack>>{track}));
        return group;
    }

}// namespace


TEST_CASE("AnimationConfig encodes, decodes and round-trips userData", "[editor]") {

    AnimationConfig config;
    CHECK(config.isDefault());

    config.autoplay = false;
    config.clip = "Walk";
    config.loop = false;
    config.speed = 1.5f;

    const auto decoded = AnimationConfig::decode(config.encode());
    REQUIRE(decoded.has_value());
    CHECK(*decoded == config);

    auto group = Group::create();

    // A default config must leave no trace in the file.
    AnimationConfig{}.write(*group);
    CHECK(group->userData.find(AnimationConfig::userDataKey) == group->userData.end());

    config.write(*group);
    const auto read = AnimationConfig::read(*group);
    REQUIRE(read.has_value());
    CHECK(*read == config);

    // Rewriting as default erases the entry again.
    AnimationConfig{}.write(*group);
    CHECK_FALSE(AnimationConfig::read(*group).has_value());
}

TEST_CASE("delimiters are stripped from encoded clip names", "[editor]") {

    AnimationConfig config;
    config.clip = "Wa;lk=Fast";
    const auto decoded = AnimationConfig::decode(config.encode());
    REQUIRE(decoded.has_value());
    CHECK(decoded->clip == "WalkFast");
}

TEST_CASE("AnimationPlaySession plays clips on objects that carry them", "[editor]") {

    auto scene = Scene::create();
    auto rig = animatedGroup("rig");
    scene->add(rig);

    AnimationPlaySession session;
    session.start(*scene);
    session.update(0.5f);

    // Halfway through the 0 -> 2 rise.
    CHECK(rig->position.y > 0.9f);
    CHECK(rig->position.y < 1.1f);

    session.stop();
}

TEST_CASE("AnimationPlaySession honours autoplay=false", "[editor]") {

    auto scene = Scene::create();
    auto still = animatedGroup("still");
    AnimationConfig config;
    config.autoplay = false;
    config.write(*still);
    scene->add(still);

    AnimationPlaySession session;
    session.start(*scene);
    session.update(0.5f);

    CHECK(still->position.y == 0.f);

    session.stop();
}
