
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

using namespace threepp;
using namespace threepp::editor;

namespace {

    const char* modeLabel(const std::string& mode) {

        if (mode == "select") return "Select";
        if (mode == "rotate") return "Rotate";
        if (mode == "scale") return "Scale";
        return "Move";
    }

}// namespace


void EditorApp::drawStatusBar() {

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    const float height = ImGui::GetFrameHeight() + 4 * s;
    statusHeight_ = height;

    ImGui::SetNextWindowPos({viewport->Pos.x, viewport->Pos.y + viewport->Size.y - height});
    ImGui::SetNextWindowSize({viewport->Size.x, height});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10 * s, 2 * s});

    if (ImGui::Begin("##status", nullptr, layout::barFlags)) {

        ImGui::Text("%.0f FPS", fps_);
        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "|");

        ImGui::SameLine();
        ImGui::Text("%zu objects", objectCount_);
        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "|");

        ImGui::SameLine();
        ImGui::Text("%s %s", viewPresetLabel(viewPreset()), orthographic() ? "ortho" : "persp");
        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "|");

        ImGui::SameLine();
        ImGui::Text("%s / %s", modeLabel(gizmoMode_), gizmoWorldSpace_ ? "World" : "Local");
        if (snapEnabled_ || ImGui::GetIO().KeyShift) {
            ImGui::SameLine();
            ImGui::TextColored(theme::accent(), "snap");
        }

        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "|");
        ImGui::SameLine();
        if (auto* selected = selection_.get()) {
            ImGui::Text("%s", selected->name.empty() ? selected->type().c_str() : selected->name.c_str());
        } else {
            ImGui::TextColored(theme::muted(), "no selection");
        }

        if (isPlaying()) {
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "|");
            ImGui::SameLine();
            ImGui::TextColored(play_.paused() ? theme::warning() : theme::playing(),
                               "%s  %.1fs", play_.paused() ? "PAUSED" : "PLAYING", play_.elapsed());
        }

        if (!statusFlash_.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "|");
            ImGui::SameLine();
            ImGui::TextColored(theme::accent(), "%s", statusFlash_.c_str());
        }

        // Right-aligned document state.
        const auto title = document_.title();
        const float width = ImGui::CalcTextSize(title.c_str()).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - width - 14 * s);
        ImGui::TextColored(document_.dirty() ? theme::warning() : theme::muted(), "%s", title.c_str());
    }
    ImGui::End();
    ImGui::PopStyleVar();
}
