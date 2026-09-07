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

        // Leave authored characters alone: their clips belong to a
        // CharacterPlaySession, which crossfades between them by name, and a
        // second mixer playing "whatever clip is first" would fight it for
        // every bone.
        //
        // Off by default, and set by the app that actually REGISTERED a
        // character session — which is a PhysX-only session. A build without
        // PhysX has nothing to defer to, and deferring anyway would leave an
        // authored character standing in its bind pose.
        void setSkipCharacters(bool skip) { skipCharacters_ = skip; }

        void start(Scene& scene) override;
        void update(float dt) override;
        void stop() override;

    private:
        std::vector<std::unique_ptr<AnimationMixer>> mixers_;
        bool skipCharacters_ = false;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ANIMATIONPLAYSESSION_HPP
