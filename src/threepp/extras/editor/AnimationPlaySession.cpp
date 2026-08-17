
#include "threepp/extras/editor/AnimationPlaySession.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/CharacterConfig.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using namespace threepp::editor;


AnimationPlaySession::AnimationPlaySession() = default;

AnimationPlaySession::~AnimationPlaySession() = default;

void AnimationPlaySession::start(Scene& scene) {

    scene.traverse([this](Object3D& object) {
        if (object.animations.empty()) return;

        // An authored character's clips belong to its own controller, which
        // crossfades between them by name. A second mixer playing "whatever
        // clip is first" over the top would fight it for every bone — so the
        // whole subtree is left to CharacterPlaySession, when there is one
        // (see setSkipCharacters).
        if (skipCharacters_) {
            for (const Object3D* o = &object; o != nullptr; o = o->parent) {
                if (CharacterConfig::isCharacter(*o)) return;
            }
        }

        const auto config = AnimationConfig::read(object).value_or(AnimationConfig{});
        if (!config.autoplay) return;

        auto clip = config.clip.empty()
                            ? object.animations.front()
                            : AnimationClip::findByName(object.animations, config.clip);
        // A renamed or re-imported clip should degrade to "play something",
        // not to a silent T-pose.
        if (!clip) clip = object.animations.front();

        auto mixer = std::make_unique<AnimationMixer>(object);
        auto* action = mixer->clipAction(clip);
        if (!action) return;

        action->setLoop(config.loop ? Loop::Repeat : Loop::Once);
        // One-shot clips hold their last frame instead of snapping back while
        // the session is still running.
        action->setClampWhenFinished(true);
        action->setEffectiveTimeScale(config.speed);
        action->play();

        mixers_.push_back(std::move(mixer));
    });
}

void AnimationPlaySession::update(float dt) {

    for (const auto& mixer : mixers_) mixer->update(dt);
}

void AnimationPlaySession::stop() {

    mixers_.clear();
}
