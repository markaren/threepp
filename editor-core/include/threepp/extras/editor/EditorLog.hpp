// The log categories of the editor document model and the play sessions.
//
// The play sessions report through threepp::log, in one category per session:
//
//   editor.physics       rigid bodies, robots, joints, vehicles, soft bodies, triggers
//   editor.character     character controllers
//   editor.granular      granular chutes
//   editor.particles     particle fields
//   editor.sensors       sensors and their recordings
//   editor.splat_surface splat scans baked into colliders
//   editor.audio         sounds and acoustics
//   editor.robot         robots revived from a saved document
//   editor.script        Python scripts (apps/editor/scripting)
//
// The editor shows them in its Console tab; the player and the tests see them
// on stderr, or wherever a sink sends them. A session's setLogger() callback
// still receives each message, after the record is written.

#ifndef THREEPP_EDITOR_EDITORLOG_HPP
#define THREEPP_EDITOR_EDITORLOG_HPP

#include "threepp/utils/Log.hpp"

#include <functional>
#include <string>

namespace threepp::editor {

    inline constinit log::Category kLogPhysics{"editor.physics"};
    inline constinit log::Category kLogCharacter{"editor.character"};
    inline constinit log::Category kLogGranular{"editor.granular"};
    inline constinit log::Category kLogParticles{"editor.particles"};
    inline constinit log::Category kLogSensors{"editor.sensors"};
    inline constinit log::Category kLogSplatSurface{"editor.splat_surface"};
    inline constinit log::Category kLogAudio{"editor.audio"};
    inline constinit log::Category kLogRobot{"editor.robot"};
    inline constinit log::Category kLogScript{"editor.script"};

    // One session message: a record in `category` at `level`, then the
    // session's own callback, if one is set.
    inline void report(const log::Category& category, log::Level level, const std::string& message,
                       const std::function<void(const std::string&)>& callback) {

        if (category.enabled(level)) log::write(category, level, message);
        if (callback) callback(message);
    }

}// namespace threepp::editor

#endif//THREEPP_EDITOR_EDITORLOG_HPP
