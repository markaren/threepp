
#ifndef THREEPP_ANIMATIONMIXER_HPP
#define THREEPP_ANIMATIONMIXER_HPP

#include "threepp/animation/AnimationAction.hpp"
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/core/Object3D.hpp"

namespace threepp {

    class AnimationMixer: public EventDispatcher {

    public:
        float time{0};
        float timeScale{1.f};

        explicit AnimationMixer(Object3D& root);

        AnimationAction* clipAction(const std::shared_ptr<AnimationClip>& clip,
                                    Object3D* optionalRoot = nullptr,
                                    std::optional<AnimationBlendMode> blendMode = std::nullopt) const;

        void stopAllAction() const;

        void update(float dt) const;

        ~AnimationMixer() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        bool _isActiveAction(AnimationAction& action) const;

        void _activateAction(const std::shared_ptr<AnimationAction>& action) const;

        void _deactivateAction(const std::shared_ptr<AnimationAction>& action) const;

        friend class AnimationAction;
    };

}// namespace threepp

#endif//THREEPP_ANIMATIONMIXER_HPP
