// Plays each object's authored animation during a play session.
//
// Any object whose `animations` vector is non-empty gets a mixer, unless its
// AnimationConfig (userData["animation"]) switched autoplay off. The editor's
// snapshot/restore puts every pose back on stop, so this session never has to
// undo anything itself.

#ifndef THREEPP_EDITOR_ANIMATIONPLAYSESSION_HPP
#define THREEPP_EDITOR_ANIMATIONPLAYSESSION_HPP

#include "threepp/extras/editor/PlaySession.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class AnimationMixer;

}

namespace threepp::editor {

    class AnimationPlaySession: public PlaySession {

    public:
        // Out of line: the mixer is forward-declared here, and the implicit
        // members would need it complete in every including TU.
        AnimationPlaySession();
        ~AnimationPlaySession() override;

        void start(Scene& scene) override;
        void update(float dt) override;
        void stop() override;

    private:
        std::vector<std::unique_ptr<AnimationMixer>> mixers_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ANIMATIONPLAYSESSION_HPP
