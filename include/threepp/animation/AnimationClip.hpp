
#ifndef THREEPP_ANIMATIONCLIP_HPP
#define THREEPP_ANIMATIONCLIP_HPP

#include "threepp/animation/KeyframeTrack.hpp"
#include "threepp/math/MathUtils.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class Object3D;
    class AnimationAction;

    class AnimationClip {

    public:
        AnimationBlendMode blendMode;

        explicit AnimationClip(std::string name,
                               float duration = 1,
                               const std::vector<std::shared_ptr<KeyframeTrack>>& tracks = {},
                               AnimationBlendMode blendMode = AnimationBlendMode::Normal);

        [[nodiscard]] std::string uuid() const;

        void resetDuration();

        static std::shared_ptr<AnimationClip> findByName(const Object3D& object, const std::string& name);
        static std::shared_ptr<AnimationClip> findByName(const std::vector<std::shared_ptr<AnimationClip>>& clipArray, const std::string& name);

    private:
        std::string uuid_{math::generateUUID()};

        std::string name;
        std::vector<std::shared_ptr<KeyframeTrack>> tracks;
        float duration;

        friend class AnimationAction;
        friend class AnimationMixer;
    };

}// namespace threepp

#endif//THREEPP_ANIMATIONCLIP_HPP
