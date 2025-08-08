
#include "threepp/animation/AnimationClip.hpp"

#include "threepp/animation/AnimationAction.hpp"
#include "threepp/core/Object3D.hpp"

#include <utility>

using namespace threepp;

AnimationClip::AnimationClip(std::string name, float duration, const std::vector<std::shared_ptr<KeyframeTrack>>& tracks, AnimationBlendMode blendMode)
    : name(std::move(name)),
      tracks(tracks),
      duration(duration),
      blendMode(blendMode) {

    // this means it should figure out its duration by scanning the tracks
    if (this->duration < 0) {

        this->resetDuration();
    }
}

std::string AnimationClip::uuid() const {

    return uuid_;
}

std::shared_ptr<AnimationClip> AnimationClip::findByName(const std::vector<std::shared_ptr<AnimationClip>>& clipArray, const std::string& name) {

    for (auto& clip : clipArray) {

        if (clip->name == name) {

            return clip;
        }
    }

    return nullptr;
}

std::shared_ptr<AnimationClip> AnimationClip::findByName(const Object3D& object, const std::string& name) {

    if (!object.geometry()) return nullptr;

    for (auto& clip : object.animations) {

        if (clip->name == name) {

            return clip;
        }
    }

    return nullptr;
}

void AnimationClip::resetDuration() {

    float duration = 0;

    for (unsigned i = 0, n = tracks.size(); i != n; ++i) {

        auto track = this->tracks[i];

        auto& times = track->getTimes();
        duration = std::max(duration, times.back());
    }

    this->duration = duration;
}
