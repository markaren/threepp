// Sensor point cloud, for the player's viewport.
//
// The player records what the sensors measured, but a CSV answers "how many"
// and never "where did the beams land" — the only question that matters when a
// scan looks wrong. The editor draws every playing vision sensor's returns as
// ONE Points, coloured green-to-red by range (apps/editor/SensorOverlay.cpp);
// this is that overlay for the player, a standalone object for the same reason
// DebugDrawOverlay is: the editor's version is welded to EditorApp.
//
// Same geometry contract as its siblings, restated because getting it wrong is
// silent: attributes rewritten IN PLACE, replaced only when the cloud outgrows
// them, orphaned geometry disposed (the renderer caches GPU buffers by
// ATTRIBUTE IDENTITY); frustumCulled and matrixAutoUpdate off, because the
// points are world-space and in-place writes never refresh cached bounds.
//
// The cloud MUST live under the core's overlay group, which the sensor session
// hides for the duration of every scan. That is not a nicety: a lidar that can
// see last frame's points builds a shell around itself and the cloud collapses
// to arm's length within a second.

#ifndef THREEPP_PLAYER_SENSORCLOUDOVERLAY_HPP
#define THREEPP_PLAYER_SENSORCLOUDOVERLAY_HPP

#include <memory>

namespace threepp {

    class Group;
    class Points;

    namespace editor {
        class SensorPlaySession;
    }

}// namespace threepp

namespace threepp::player {

    class SensorCloudOverlay {

    public:
        // `parent` is the player's overlay group (editor-only, hidden during
        // scans); `session` is borrowed and must outlive this object — in the
        // player both belong to PlayerCore, which the app destroys last.
        SensorCloudOverlay(Group& parent, editor::SensorPlaySession& session);
        ~SensorCloudOverlay();

        SensorCloudOverlay(const SensorCloudOverlay&) = delete;
        SensorCloudOverlay& operator=(const SensorCloudOverlay&) = delete;

        // Rewrite the cloud from the session's current returns. A frame where
        // nothing scanned keeps the last cloud (sensors tick on their own
        // rates, slower than the frame — blinking would just track the beat).
        void sync();

        // Take the node down (the episode ended; entries are gone).
        void clear();

    private:
        Group& parent_;
        editor::SensorPlaySession& session_;
        std::shared_ptr<Points> cloud_;
        int capacity_ = 0;
    };

}// namespace threepp::player

#endif//THREEPP_PLAYER_SENSORCLOUDOVERLAY_HPP
