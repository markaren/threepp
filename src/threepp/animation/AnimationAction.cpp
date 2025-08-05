
#include "threepp/animation/AnimationAction.hpp"

#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/animation/PropertyMixer.hpp"

using namespace threepp;

AnimationAction::AnimationAction(AnimationMixer& mixer,
                                 const std::shared_ptr<AnimationClip>& clip,
                                 Object3D* localRoot,
                                 std::optional<AnimationBlendMode> blendMode)
    : _mixer(mixer),
      _clip(clip),
      _localRoot(localRoot),
      blendMode(blendMode.value_or(clip->blendMode)),
      _propertyBindings(clip->tracks.size()) {

    auto& tracks = clip->tracks;
    const auto nTracks = tracks.size();
    std::vector<std::shared_ptr<Interpolant>> interpolants(nTracks);

    _interpolantSettings = {
            Ending::ZeroCurvature,
            Ending::ZeroCurvature};

    for (auto i = 0; i != nTracks; ++i) {

        auto interpolant = tracks[i]->createInterpolant(nullptr);
        interpolant->settings = &_interpolantSettings;
        interpolants[i] = std::move(interpolant);
    }

    this->_interpolants = interpolants;
}

AnimationAction& AnimationAction::play() {

    _mixer._activateAction(shared_from_this());

    return *this;
}

AnimationAction& AnimationAction::stop() {

    this->_mixer._deactivateAction(shared_from_this());

    return this->reset();
}

AnimationAction& AnimationAction::reset() {

    this->paused = false;
    this->enabled = true;

    this->time = 0;                 // restart clip
    this->_loopCount = -1;          // forget previous loops
    this->_startTime = std::nullopt;// forget scheduling

    return this->stopFading().stopWarping();
}

bool AnimationAction::isRunning() {

    return this->enabled && !this->paused && this->timeScale != 0 &&
           !this->_startTime && this->_mixer._isActiveAction(*this);
}

bool AnimationAction::isScheduled() {

    return this->_mixer._isActiveAction(*this);
}

AnimationAction& AnimationAction::startAt(float time) {

    this->_startTime = time;

    return *this;
}

AnimationAction& AnimationAction::setLoop(Loop mode, int repetitions) {

    this->loop = mode;
    this->repetitions = repetitions;

    return *this;
}

AnimationAction& AnimationAction::setEffectiveWeight(float weight) {

    this->weight = weight;

    // note: same logic as when updated at runtime
    this->_effectiveWeight = this->enabled ? weight : 0;

    return this->stopFading();
}

float AnimationAction::getEffectiveWeight() const {

    return this->_effectiveWeight;
}

AnimationAction& AnimationAction::stopFading() {

    const auto weightInterpolant = this->_weightInterpolant;

    if (weightInterpolant) {

        // TODO

        // this->_weightInterpolant = nullptr;
        // this->_mixer._takeBackControlInterpolant(weightInterpolant);
    }

    return *this;
}

AnimationAction& AnimationAction::setEffectiveTimeScale(float timeScale) {

    this->timeScale = timeScale;
    this->_effectiveTimeScale = this->paused ? 0 : timeScale;

    return this->stopWarping();
}

float AnimationAction::getEffectiveTimeScale() const {

    return this->_effectiveTimeScale;
}

AnimationAction& AnimationAction::setDuration(float duration) {

    this->timeScale = this->_clip->duration / duration;

    return this->stopWarping();
}

AnimationAction& AnimationAction::syncWith(AnimationAction& action) {

    this->time = action.time;
    this->timeScale = action.timeScale;

    return this->stopWarping();
}

AnimationAction& AnimationAction::halt(float duration) {

    return this->warp(this->_effectiveTimeScale, 0, duration);
}

AnimationAction& AnimationAction::warp(float startTimeScale, float endTimeScale, float duration) {

    auto& mixer = this->_mixer;
    const auto now = mixer.time;
    const auto timeScale = this->timeScale;

    const auto& interpolant = this->_timeScaleInterpolant;

    if (!interpolant) {

        // interpolant = mixer._lendControlInterpolant();
        //        this->_timeScaleInterpolant = interpolant;
    }

    auto& times = interpolant->parameterPositions;
    auto& values = interpolant->sampleValues;

    times[0] = now;
    times[1] = now + duration;

    values[0] = startTimeScale / timeScale;
    values[1] = endTimeScale / timeScale;

    return *this;
}

AnimationAction& AnimationAction::stopWarping() {

    const auto& timeScaleInterpolant = this->_timeScaleInterpolant;

    if (timeScaleInterpolant) {

        //        this->_timeScaleInterpolant = nullptr;
        //        this->_mixer._takeBackControlInterpolant(timeScaleInterpolant);
    }

    return *this;
}

void AnimationAction::_update(float time, float deltaTime, int timeDirection, int accuIndex) {

    // called by the mixer

    if (!this->enabled) {

        // call ._updateWeight() to update ._effectiveWeight

        this->_updateWeight(time);
        return;
    }

    const auto startTime = this->_startTime;

    if (startTime) {

        // check for scheduled start of action

        const auto timeRunning = (time - *startTime) * static_cast<float>(timeDirection);
        if (timeRunning < 0 || timeDirection == 0) {

            return;// yet to come / don't decide when delta = 0
        }

        // start

        this->_startTime = std::nullopt;// unschedule
        deltaTime = static_cast<float>(timeDirection) * timeRunning;
    }

    // apply time scale and advance time

    deltaTime *= this->_updateTimeScale(time);
    const auto clipTime = this->_updateTime(deltaTime);

    // note: _updateTime may disable the action resulting in
    // an effective weight of 0

    const auto weight = this->_updateWeight(time);

    if (weight > 0) {

        const auto& interpolants = this->_interpolants;
        const auto& propertyMixers = this->_propertyBindings;

        switch (this->blendMode) {

            case AnimationBlendMode::Additive:

                for (unsigned j = 0, m = interpolants.size(); j != m; ++j) {

                    interpolants[j]->evaluate(clipTime);
                    propertyMixers[j]->accumulateAdditive(weight);
                }

                break;

            case AnimationBlendMode::Normal:
            default:

                for (unsigned j = 0, m = interpolants.size(); j != m; ++j) {

                    interpolants[j]->evaluate(clipTime);
                    propertyMixers[j]->accumulate(accuIndex, weight);
                }
        }
    }
}

float AnimationAction::_updateWeight(float time) {

    float weight = 0;

    if (this->enabled) {

        weight = this->weight;
        const auto& interpolant = this->_weightInterpolant;

        if (interpolant) {

            const auto interpolantValue = interpolant->evaluate(time)[0];

            weight *= interpolantValue;

            if (time > interpolant->parameterPositions[1]) {

                this->stopFading();

                if (interpolantValue == 0) {

                    // faded out, disable
                    this->enabled = false;
                }
            }
        }
    }

    this->_effectiveWeight = weight;
    return weight;
}

float AnimationAction::_updateTimeScale(float time) {

    float timeScale = 0;

    if (!this->paused) {

        timeScale = this->timeScale;

        const auto& interpolant = this->_timeScaleInterpolant;

        if (interpolant) {

            const auto interpolantValue = interpolant->evaluate(time)[0];

            timeScale *= interpolantValue;

            if (time > interpolant->parameterPositions[1]) {

                this->stopWarping();

                if (timeScale == 0) {

                    // motion has halted, pause
                    this->paused = true;

                } else {

                    // warp done - apply final time scale
                    this->timeScale = timeScale;
                }
            }
        }
    }

    this->_effectiveTimeScale = timeScale;
    return timeScale;
}

float AnimationAction::_updateTime(float deltaTime) {

    const auto duration = this->_clip->duration;
    const auto loop = this->loop;

    float time = this->time + deltaTime;
    int loopCount = this->_loopCount;

    const bool pingPong = (loop == Loop::PingPong);

    if (deltaTime == 0) {
        if (loopCount == -1) return time;
        return (pingPong && (loopCount & 1) == 1) ? duration - time : time;
    }

    if (loop == Loop::Once) {
        if (loopCount == -1) {
            // just started
            this->_loopCount = 0;
            this->_setEndings(true, true, false);
        }

        if (time >= duration) {
            time = duration;
        } else if (time < 0) {
            time = 0;
        } else {
            this->time = time;
            return time;
        }

        if (this->clampWhenFinished) this->paused = true;
        else
            this->enabled = false;

        this->time = time;

        // Dispatch finished event if needed
        // this->_mixer.dispatchEvent({ ... });

        return time;
    }

    // Repeat or PingPong
    if (loopCount == -1) {
        // just started
        if (deltaTime >= 0) {
            loopCount = 0;
            this->_setEndings(true, this->repetitions == 0, pingPong);
        } else {
            this->_setEndings(this->repetitions == 0, true, pingPong);
        }
    }

    if (time >= duration || time < 0) {
        // wrap around
        int loopDelta = static_cast<int>(std::floor(time / duration));
        time -= duration * loopDelta;
        loopCount += std::abs(loopDelta);

        int pending = (this->repetitions == -1) ? 1 : (this->repetitions - loopCount);

        if (this->repetitions != -1 && pending <= 0) {
            if (this->clampWhenFinished) this->paused = true;
            else
                this->enabled = false;

            time = deltaTime > 0 ? duration : 0;
            this->time = time;

            // Dispatch finished event if needed
            // this->_mixer.dispatchEvent({ ... });

            return time;
        } else {
            if (pending == 1) {
                bool atStart = deltaTime < 0;
                this->_setEndings(atStart, !atStart, pingPong);
            } else {
                this->_setEndings(false, false, pingPong);
            }

            this->_loopCount = loopCount;
            this->time = time;

            // Dispatch loop event if needed
            // this->_mixer.dispatchEvent({ ... });
        }
    } else {
        this->time = time;
    }

    if (pingPong && (loopCount & 1) == 1) {
        return duration - time;
    }

    return time;
}

void AnimationAction::_setEndings(bool atStart, bool atEnd, bool pingPong) {

    auto& settings = this->_interpolantSettings;

    if (pingPong) {

        settings.endingStart = Ending::ZeroSlope;
        settings.endingEnd = Ending::ZeroSlope;

    } else {

        // assuming for LoopOnce atStart == atEnd == true

        if (atStart) {

            settings.endingStart = this->zeroSlopeAtStart ? Ending::ZeroSlope : Ending::ZeroCurvature;

        } else {

            settings.endingStart = Ending::WrapAround;
        }

        if (atEnd) {

            settings.endingEnd = this->zeroSlopeAtEnd ? Ending::ZeroSlope : Ending::ZeroCurvature;

        } else {

            settings.endingEnd = Ending::WrapAround;
        }
    }
}
