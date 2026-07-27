
#include "EditorTheme.hpp"

#include <imgui.h>

namespace {

    // One accent (a desaturated cyan-blue), everything else neutral grey. The
    // greys are picked so that panel < window < frame < hovered reads as a
    // consistent depth order at a glance.
    constexpr ImVec4 kAccent{0.26f, 0.59f, 0.98f, 1.00f};
    constexpr ImVec4 kAccentDim{0.26f, 0.59f, 0.98f, 0.45f};
    constexpr ImVec4 kWarning{0.95f, 0.75f, 0.30f, 1.00f};
    constexpr ImVec4 kDanger{0.90f, 0.42f, 0.40f, 1.00f};
    constexpr ImVec4 kMuted{0.55f, 0.57f, 0.60f, 1.00f};
    constexpr ImVec4 kPlaying{0.35f, 0.82f, 0.45f, 1.00f};
    // Viewport marker icons at rest: bright enough to read against both the
    // dark grid and lit geometry, dim enough that the accent still stands out
    // when the object is selected.
    constexpr ImVec4 kMarkerIdle{0.78f, 0.80f, 0.84f, 1.00f};

    constexpr ImVec4 grey(float v, float a = 1.f) { return {v, v, v, a}; }

}// namespace


ImVec4 threepp::editor::theme::accent() { return kAccent; }
ImVec4 threepp::editor::theme::warning() { return kWarning; }
ImVec4 threepp::editor::theme::danger() { return kDanger; }
ImVec4 threepp::editor::theme::muted() { return kMuted; }
ImVec4 threepp::editor::theme::playing() { return kPlaying; }
ImVec4 threepp::editor::theme::markerIdle() { return kMarkerIdle; }

void threepp::editor::theme::apply(float scale) {

    ImGuiStyle& style = ImGui::GetStyle();

    // --- metrics ------------------------------------------------------------
    // Small radii everywhere (a hint of softness, not a toy), generous vertical
    // item spacing so dense inspector rows stay readable, and no window border
    // — the panels tile the screen, borders between them would double up.
    style.WindowPadding = {10 * scale, 10 * scale};
    style.FramePadding = {8 * scale, 4 * scale};
    style.CellPadding = {6 * scale, 4 * scale};
    style.ItemSpacing = {8 * scale, 6 * scale};
    style.ItemInnerSpacing = {6 * scale, 5 * scale};
    style.IndentSpacing = 18 * scale;
    style.ScrollbarSize = 12 * scale;
    style.GrabMinSize = 10 * scale;

    style.WindowBorderSize = 0;
    style.ChildBorderSize = 1 * scale;
    style.PopupBorderSize = 1 * scale;
    style.FrameBorderSize = 0;
    style.TabBorderSize = 0;

    style.WindowRounding = 0;
    style.ChildRounding = 4 * scale;
    style.FrameRounding = 4 * scale;
    style.PopupRounding = 4 * scale;
    style.ScrollbarRounding = 6 * scale;
    style.GrabRounding = 4 * scale;
    style.TabRounding = 4 * scale;

    style.WindowTitleAlign = {0.0f, 0.5f};
    style.WindowMenuButtonPosition = ImGuiDir_None;
    style.SeparatorTextBorderSize = 1 * scale;
    style.SeparatorTextPadding = {16 * scale, 4 * scale};

    // --- colours ------------------------------------------------------------
    ImVec4* c = style.Colors;

    c[ImGuiCol_Text] = grey(0.88f);
    c[ImGuiCol_TextDisabled] = kMuted;

    c[ImGuiCol_WindowBg] = grey(0.13f);
    c[ImGuiCol_ChildBg] = grey(0.11f);
    c[ImGuiCol_PopupBg] = grey(0.15f, 0.98f);
    c[ImGuiCol_MenuBarBg] = grey(0.16f);

    c[ImGuiCol_Border] = grey(0.24f);
    c[ImGuiCol_BorderShadow] = grey(0.f, 0.f);

    c[ImGuiCol_FrameBg] = grey(0.20f);
    c[ImGuiCol_FrameBgHovered] = grey(0.26f);
    c[ImGuiCol_FrameBgActive] = grey(0.30f);

    c[ImGuiCol_TitleBg] = grey(0.10f);
    c[ImGuiCol_TitleBgActive] = grey(0.16f);
    c[ImGuiCol_TitleBgCollapsed] = grey(0.10f);

    c[ImGuiCol_ScrollbarBg] = grey(0.11f);
    c[ImGuiCol_ScrollbarGrab] = grey(0.28f);
    c[ImGuiCol_ScrollbarGrabHovered] = grey(0.35f);
    c[ImGuiCol_ScrollbarGrabActive] = grey(0.42f);

    c[ImGuiCol_CheckMark] = kAccent;
    c[ImGuiCol_SliderGrab] = {0.35f, 0.55f, 0.80f, 1.00f};
    c[ImGuiCol_SliderGrabActive] = kAccent;

    c[ImGuiCol_Button] = grey(0.22f);
    c[ImGuiCol_ButtonHovered] = grey(0.30f);
    c[ImGuiCol_ButtonActive] = kAccentDim;

    c[ImGuiCol_Header] = grey(0.24f);
    c[ImGuiCol_HeaderHovered] = grey(0.30f);
    c[ImGuiCol_HeaderActive] = kAccentDim;

    c[ImGuiCol_Separator] = grey(0.24f);
    c[ImGuiCol_SeparatorHovered] = kAccentDim;
    c[ImGuiCol_SeparatorActive] = kAccent;

    c[ImGuiCol_ResizeGrip] = grey(0.f, 0.f);
    c[ImGuiCol_ResizeGripHovered] = kAccentDim;
    c[ImGuiCol_ResizeGripActive] = kAccent;

    c[ImGuiCol_Tab] = grey(0.16f);
    c[ImGuiCol_TabHovered] = grey(0.28f);
    c[ImGuiCol_TabSelected] = grey(0.24f);
    c[ImGuiCol_TabSelectedOverline] = kAccent;
    c[ImGuiCol_TabDimmed] = grey(0.13f);
    c[ImGuiCol_TabDimmedSelected] = grey(0.20f);

    c[ImGuiCol_TableHeaderBg] = grey(0.18f);
    c[ImGuiCol_TableBorderStrong] = grey(0.26f);
    c[ImGuiCol_TableBorderLight] = grey(0.20f);
    c[ImGuiCol_TableRowBg] = grey(0.f, 0.f);
    c[ImGuiCol_TableRowBgAlt] = grey(1.f, 0.03f);

    c[ImGuiCol_TextSelectedBg] = kAccentDim;
    c[ImGuiCol_DragDropTarget] = kWarning;
    c[ImGuiCol_NavCursor] = kAccent;
    c[ImGuiCol_ModalWindowDimBg] = grey(0.05f, 0.65f);
}
