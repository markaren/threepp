
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

void EditorApp::drawPlayBanner() {

    if (!isPlaying()) return;

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    // A slim strip across the viewport, high enough to be unmissable and out of
    // the way of the toolbar.
    const char* text = play_.paused() ? "PAUSED" : "PLAY";
    const ImVec2 size = ImGui::CalcTextSize(text);
    const float width = size.x + 40 * s;
    const float height = size.y + 10 * s;

    ImGui::SetNextWindowPos({viewport->Pos.x + (viewport->Size.x - width) * 0.5f,
                             viewport->Pos.y + menuHeight_ + toolbarHeight_ + 10 * s});
    ImGui::SetNextWindowSize({width, height});
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg,
                          play_.paused() ? ImVec4{0.35f, 0.28f, 0.10f, 1.f}
                                         : ImVec4{0.10f, 0.32f, 0.16f, 1.f});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {20 * s, 5 * s});

    if (ImGui::Begin("##playBanner", nullptr,
                     layout::barFlags | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoFocusOnAppearing)) {
        ImGui::TextColored(play_.paused() ? theme::warning() : theme::playing(), "%s", text);
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}
