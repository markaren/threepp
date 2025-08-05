
#ifndef THREEPP_ANIMATIONACTION_HPP
#define THREEPP_ANIMATIONACTION_HPP

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/math/Interpolant.hpp"
#include "threepp/constants.hpp"

namespace threepp {

    class AnimationMixer;
    class PropertyMixer;

    class AnimationAction: public std::enable_shared_from_this<AnimationAction> {

    public:
        AnimationBlendMode blendMode;

        AnimationAction(AnimationMixer& mixer,
                        const std::shared_ptr<AnimationClip>& clip,
                        Object3D* localRoot = nullptr,
                        std::optional<AnimationBlendMode> blendMode = std::nullopt);

        // State & Scheduling

        AnimationAction& play();

        AnimationAction& stop();

        AnimationAction& reset();

        bool isRunning();

        // return true when play has been called
        bool isScheduled();

        AnimationAction& startAt(float time);

        AnimationAction& setLoop(Loop mode, int repetitions = -1);

        // Weight

        // set the weight stopping any scheduled fading
        // although .enabled = false yields an effective weight of zero, this
        // method does *not* change .enabled, because it would be confusing
        AnimationAction& setEffectiveWeight(float weight);

        // return the weight considering fading and .enabled
        [[nodiscard]] float getEffectiveWeight() const;

        AnimationAction& stopFading();

        // Time Scale Control

        // set the time scale stopping any scheduled warping
        // although .paused = true yields an effective time scale of zero, this
        // method does *not* change .paused, because it would be confusing
        AnimationAction& setEffectiveTimeScale(float timeScale);

        // return the time scale considering warping and .paused
        float getEffectiveTimeScale() const;

        AnimationAction& setDuration(float duration);

        AnimationAction& syncWith(AnimationAction& action);

        AnimationAction& halt(float duration);

        AnimationAction& warp(float startTimeScale, float endTimeScale, float duration);

        AnimationAction& stopWarping();


    private:
        AnimationMixer& _mixer;
        std::shared_ptr<AnimationClip> _clip;
        Object3D* _localRoot;

        InterpolantSettings _interpolantSettings;
        std::vector<std::shared_ptr<Interpolant>> _interpolants;

        std::vector<std::shared_ptr<PropertyMixer>> _propertyBindings;

        std::optional<size_t> _cacheIndex;      // for the memory manager
        std::optional<size_t> _byClipCacheIndex;// for the memory manager

        std::shared_ptr<Interpolant> _timeScaleInterpolant;
        std::shared_ptr<Interpolant> _weightInterpolant;

        Loop loop{Loop::Repeat};
        int _loopCount = -1;

        // global mixer time when the action is to be started
        // it's set back to 'null' upon start of the action
        std::optional<float> _startTime;

        // scaled local time of the action
        // gets clamped or wrapped to 0..clip.duration according to loop
        float time = 0;

        float timeScale = 1;
        float _effectiveTimeScale = 1;

        float weight = 1;
        float _effectiveWeight = 1;

        int repetitions = -1;// no. of repetitions when looping

        bool paused = false;// true -> zero effective time scale
        bool enabled = true;// false -> zero effective weight

        bool clampWhenFinished = false;// keep feeding the last frame?

        bool zeroSlopeAtStart = true;// for smooth interpolation w/o separate
        bool zeroSlopeAtEnd = true;  // clips for start, loop and end

        void _update(float time, float deltaTime, int timeDirection, int accuIndex);

        float _updateWeight(float time);

        float _updateTimeScale(float time);

        float _updateTime(float deltaTime);

        void _setEndings(bool atStart, bool atEnd, bool pingPong);

        friend class AnimationMixer;
    };
}// namespace threepp

#endif//THREEPP_ANIMATIONACTION_HPP
