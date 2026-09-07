// Shared geometry for the fixed panel layout.
//
// The editor does not use ImGui docking (the vendored copy is the master
// branch). Instead every panel is positioned and sized from the viewport each
// frame, which gives a stable, predictable layout and costs nothing.

#ifndef THREEPP_EDITOR_PANELLAYOUT_HPP
#define THREEPP_EDITOR_PANELLAYOUT_HPP

#include <imgui.h>

namespace threepp::editor::layout {

    // Every panel size is user-draggable and lives in EditorSettings rather
    // than here — see EditorApp::hierarchyPx() / inspectorPx() /
    // bottomPanelPx(), which apply the monitor content scale.

    // Grab strip between a panel and the viewport.
    inline constexpr float splitterThickness = 6.f;

    inline constexpr ImGuiWindowFlags panelFlags =
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoSavedSettings;

    inline constexpr ImGuiWindowFlags barFlags = panelFlags | ImGuiWindowFlags_NoTitleBar |
                                                 ImGuiWindowFlags_NoScrollbar |
                                                 ImGuiWindowFlags_NoScrollWithMouse;

}// namespace threepp::editor::layout

#endif//THREEPP_EDITOR_PANELLAYOUT_HPP
