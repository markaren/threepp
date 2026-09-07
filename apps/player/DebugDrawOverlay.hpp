// Script debug draw, for the player's viewport.
//
// The same seam and the same geometry contract as the editor's
// (apps/editor/DebugDrawOverlay.cpp) — threepp.editor.draw_line and friends land
// in scripting::debugDraw() as world-space segments, and this drains the list
// into ONE LineSegments each rendered frame. A standalone object rather than a
// member of the app, because the player's core needs to hand somebody a drain
// callback and the editor's version is welded to EditorApp.
//
// The rules that matter, restated because getting them wrong is silent:
//
//   * the attributes are rewritten IN PLACE and replaced only when outgrown,
//     and the orphaned geometry is disposed when they are. The renderer caches
//     GPU buffers by ATTRIBUTE IDENTITY, so a fresh attribute every frame can
//     hand it a recycled pointer that reads as already uploaded;
//   * frustumCulled and matrixAutoUpdate are off, because the segments are
//     already world-space and in-place updates never refresh cached bounds;
//   * depth test AND depth write off — debugging wants to see the ray that ends
//     inside a mesh, and the lines must not shadow anything drawn after them.
//
// Immediate mode: drained is gone. A script that wants a line to persist draws
// it again next update(), which it is called every frame to do.

#ifndef THREEPP_PLAYER_DEBUGDRAWOVERLAY_HPP
#define THREEPP_PLAYER_DEBUGDRAWOVERLAY_HPP

#include <functional>
#include <memory>
#include <string>

namespace threepp {

    class Group;
    class LineSegments;

}// namespace threepp

namespace threepp::player {

    class DebugDrawOverlay {

    public:
        // `parent` is the player's overlay group — editor-only, and hidden for
        // the duration of every sensor scan, so a lidar can never range against
        // one of these lines.
        explicit DebugDrawOverlay(Group& parent);
        ~DebugDrawOverlay();

        DebugDrawOverlay(const DebugDrawOverlay&) = delete;
        DebugDrawOverlay& operator=(const DebugDrawOverlay&) = delete;

        void setLogger(std::function<void(const std::string&)> logger);

        // Drain the list into the lines. This is what PlayerCore calls as its
        // debug-draw drain, so consuming the segments and emptying the list are
        // the same act.
        void sync();

        // Take the node down (the session went away).
        void clear();

    private:
        Group& parent_;
        std::shared_ptr<LineSegments> lines_;
        std::function<void(const std::string&)> logger_;
        int capacity_ = 0;
        bool warned_ = false;
    };

}// namespace threepp::player

#endif//THREEPP_PLAYER_DEBUGDRAWOVERLAY_HPP
