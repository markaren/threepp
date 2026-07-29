// The Sensors tab: what the rig is measuring, right now.
//
// The inspector says what was AUTHORED. This says what came back — and the two
// disagreeing is the normal case when a rig is wrong: an IMU on an object with no
// rigid body, a LIDAR whose near plane swallows the robot it is mounted on, a
// depth camera facing into a wall. Every one of those authors cleanly and
// produces nothing, so a panel that only listed the configuration would show a
// perfectly healthy rig measuring air.
//
// The plots are ImGui::PlotLines over the session's own ring buffers. Vendored
// imgui only — no plotting dependency — which is enough for the question a plot
// answers here: is this channel alive, and does its shape look like physics.
// The session owns those rings because it owns the drain: drain() empties the
// sensor, so whoever calls it is the only party that can feed a plot.
//
// Recording is a toggle and a directory. One CSV per sensor, flushed on Stop.

#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"

#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/scenes/Scene.hpp"

#include <cfloat>
#include <cstddef>
#include <string>

using namespace threepp;
using namespace threepp::editor;


void EditorApp::drawSensorsTab() {

    // How many sensors the DOCUMENT carries, whether or not anything is playing.
    // Answers "did my authoring take" before Play is ever pressed.
    std::size_t authored = 0;
    document_.scene().traverse([&](Object3D& object) {
        if (document_.isEditorOnly(object)) return;
        if (const auto config = SensorConfig::read(object); config && config->enabled) ++authored;
    });

#ifndef THREEPP_EDITOR_WITH_PHYSX
    // The session still runs the vision sensors; the body/joint entries carry a
    // per-sensor status at Play, so this is the only build-wide note needed.
    ImGui::TextColored(theme::muted(),
                       "Built without PhysX - body and joint sensors are authored, not run.");
#endif

    const float s = contentScale_;

    // --- recording ---------------------------------------------------------
    bool recording = sensors_ && sensors_->recording();
    if (ImGui::Checkbox("Record", &recording)) {
        // Re-query rather than trusting `recording`: the checkbox's own handler
        // is the only writer, but the session may not exist at all.
        if (sensors_) sensors_->setRecording(recording);
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Folder...")) {
        pendingDialog_ = PendingDialog::RecordDir;
        // The browser has no directory mode; a Save dialog's PARENT directory is
        // what gets used, and the name the user types is discarded.
        fileBrowser_.open("Recording Folder", FileBrowser::Mode::Save,
                          sensors_ ? sensors_->recordDirectory()
                                   : std::filesystem::temp_directory_path(),
                          {}, "sensors");
    }
    if (sensors_) {
        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "%s", sensors_->recordDirectory().string().c_str());
        if (sensors_->recording()) {
            ImGui::TextColored(theme::playing(), "recording - %zu rows",
                               sensors_->recordedRows());
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "(flushed on Stop)");
        }
    }

    ImGui::Separator();

    if (!isPlaying()) {
        if (authored == 0) {
            ImGui::TextColored(theme::muted(),
                               "No sensors authored. Select an object and add one in the "
                               "inspector's Sensor section.");
        } else {
            ImGui::Text("%zu sensor%s authored", authored, authored == 1 ? "" : "s");
            ImGui::TextColored(theme::muted(),
                               "Press Play to build and run them. Sensors are rebuilt from the "
                               "authored seed every Play, so a run replays exactly.");
        }
        return;
    }

    if (!sensors_ || sensors_->sensorCount() == 0) {
        ImGui::TextColored(theme::muted(), "Playing, but no sensors in this scene.");
        return;
    }

    ImGui::Text("%zu of %zu measuring", sensors_->liveCount(), sensors_->sensorCount());
    ImGui::SameLine();
    ImGui::TextColored(theme::muted(), "- sim time %.3f s", sensors_->simTime());

    if (!ImGui::BeginChild("##sensors", {0, 0})) {
        ImGui::EndChild();
        return;
    }

    int index = 0;
    for (const auto& entry : sensors_->entries()) {

        ImGui::PushID(index++);

        const char* kind = SensorConfig::label(entry->config.type);
        // Open by default: the readout exists to be read, and a rig has a
        // handful of sensors, not a hundred.
        const bool open = ImGui::TreeNodeEx("##sensor", ImGuiTreeNodeFlags_DefaultOpen,
                                            "%s  -  %s", entry->label.c_str(), kind);

        if (!entry->status.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(theme::warning(), "  (%s)", entry->status.c_str());
        }

        if (open) {
            if (entry->config.rateHz > 0.f) {
                ImGui::TextColored(theme::muted(), "%.1f Hz, seed %d",
                                   entry->config.rateHz, entry->config.seed);
            } else {
                ImGui::TextColored(theme::muted(), "every substep, seed %d", entry->config.seed);
            }

            if (SensorConfig::isVision(entry->config.type)) {
                ImGui::Text("%zu scans, %zu points, t = %.3f s",
                            entry->scans, entry->pointCount(), entry->lastTime);
                if (entry->scans > 0 && entry->pointCount() == 0) {
                    ImGui::TextColored(theme::warning(),
                                       "Scanning, but nothing in range - check the near/far "
                                       "planes and where it is pointing.");
                }
            } else {
                ImGui::Text("%zu samples, t = %.3f s", entry->samples, entry->lastTime);
            }

            if (entry->config.type == SensorConfig::Type::Contact) {
                ImGui::SameLine();
                if (entry->inContact) {
                    ImGui::TextColored(theme::playing(), "- TOUCHING (%.1f N)", entry->contactForce);
                } else {
                    ImGui::TextColored(theme::muted(), "- not touching");
                }
            }

            // --- plots -----------------------------------------------------
            for (int channel = 0; channel < entry->traceCount; ++channel) {
                const auto& trace = entry->traces[static_cast<std::size_t>(channel)];
                const char* label = entry->traceNames[static_cast<std::size_t>(channel)];
                if (!label || trace.count == 0) continue;
                // FLT_MAX on both bounds is PlotLines' autoscale. A fixed scale
                // would flatten a gyro trace whenever an accel trace shares the
                // panel, and the shape is the whole information here.
                ImGui::PlotLines(label, trace.values.data(), trace.count, trace.offset,
                                 nullptr, FLT_MAX, FLT_MAX, ImVec2(0.f, 40.f * s));
            }
            if (entry->traceCount == 0) {
                ImGui::TextColored(theme::muted(), "No scalar channels to plot.");
            }

            if (entry->rows > 0) {
                ImGui::TextColored(theme::muted(), "recording %zu rows -> %s", entry->rows,
                                   entry->csvPath.filename().string().c_str());
            }

            ImGui::TreePop();
        }

        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::EndChild();
}
