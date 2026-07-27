
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A button that stays visibly pressed while its mode is the active one.
    bool modeButton(const char* label, bool active, const ImVec2& size) {

        const bool pushed = active;
        if (pushed) {
            ImGui::PushStyleColor(ImGuiCol_Button, threepp::editor::theme::accent());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, threepp::editor::theme::accent());
        }
        const bool clicked = ImGui::Button(label, size);
        if (pushed) ImGui::PopStyleColor(2);
        return clicked;
    }

}// namespace

void EditorApp::drawToolbar() {

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    const float height = ImGui::GetFrameHeight() + ImGui::GetStyle().WindowPadding.y * 2.f;
    toolbarHeight_ = height;

    ImGui::SetNextWindowPos({viewport->Pos.x, viewport->Pos.y + menuHeight_});
    ImGui::SetNextWindowSize({viewport->Size.x, height});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10 * s, 5 * s});

    if (ImGui::Begin("##toolbar", nullptr, layout::barFlags)) {

        const ImVec2 button{86 * s, 0};

        if (modeButton("Select", gizmoMode_ == "select", button)) {
            gizmoMode_ = "select";
            applyGizmoMode();
        }
        ImGui::SameLine();
        if (modeButton("Move", gizmoMode_ == "translate", button)) {
            gizmoMode_ = "translate";
            applyGizmoMode();
        }
        ImGui::SameLine();
        if (modeButton("Rotate", gizmoMode_ == "rotate", button)) {
            gizmoMode_ = "rotate";
            applyGizmoMode();
        }
        ImGui::SameLine();
        if (modeButton("Scale", gizmoMode_ == "scale", button)) {
            gizmoMode_ = "scale";
            applyGizmoMode();
        }

        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "|");
        ImGui::SameLine();

        if (modeButton(gizmoWorldSpace_ ? "World" : "Local", false, {76 * s, 0})) {
            gizmoWorldSpace_ = !gizmoWorldSpace_;
            applyGizmoMode();
        }
        ImGui::SameLine();
        if (modeButton("Snap", snapEnabled_, {76 * s, 0})) {
            snapEnabled_ = !snapEnabled_;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Snap translate/rotate/scale to a grid.\nHold Shift for the same effect.");
        }

        // --- transport, centred ---------------------------------------------
        const float transportWidth = 3 * 46 * s + 2 * ImGui::GetStyle().ItemSpacing.x;
        ImGui::SameLine();
        ImGui::SetCursorPosX((viewport->Size.x - transportWidth) * 0.5f);

        const bool playing = isPlaying();
        const ImVec2 transport{46 * s, 0};

        if (playing && !play_.paused()) {
            ImGui::PushStyleColor(ImGuiCol_Button, theme::playing());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, theme::playing());
        }
        if (ImGui::Button("Play", transport)) startPlay();
        if (playing && !play_.paused()) ImGui::PopStyleColor(2);

        ImGui::SameLine();
        if (!playing) ImGui::BeginDisabled();
        if (modeButton("Pause", playing && play_.paused(), transport)) togglePause();
        ImGui::SameLine();
        if (ImGui::Button("Stop", transport)) stopPlay();
        if (!playing) ImGui::EndDisabled();
    }
    ImGui::End();
    ImGui::PopStyleVar();
}
