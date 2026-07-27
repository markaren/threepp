
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"

#include "threepp/scenes/Scene.hpp"

#include <filesystem>

using namespace threepp;
using namespace threepp::editor;

void EditorApp::drawMenuBar() {

    // The renderer-settings panel is the one piece of editor UI that already
    // existed in threepp — reused wholesale rather than reimplemented.
    static bool showRendererSettings = false;
    static bool showShortcuts = false;

    if (ImGui::BeginMainMenuBar()) {

        menuHeight_ = ImGui::GetWindowSize().y;

        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("New", "Ctrl+N")) pendingAction_ = PendingAction::New;
            if (ImGui::MenuItem("Open...", "Ctrl+O")) pendingAction_ = PendingAction::Open;

            if (ImGui::BeginMenu("Open Recent", !settings_.recentFiles().empty())) {
                for (const auto& recent : settings_.recentFiles()) {
                    if (ImGui::MenuItem(recent.c_str())) {
                        if (document_.dirty()) {
                            pendingAction_ = PendingAction::OpenPath;
                            pendingPath_ = recent;
                        } else {
                            openScene(recent);
                        }
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Clear")) settings_.clearRecentFiles();
                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Save", "Ctrl+S")) saveScene();
            if (ImGui::MenuItem("Save As...")) {
                pendingDialog_ = PendingDialog::SaveAs;
                fileBrowser_.open("Save Scene As", FileBrowser::Mode::Save,
                                  settings_.sceneDir, {".json"},
                                  document_.hasPath() ? document_.path().filename().string()
                                                      : std::string("scene.json"));
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Import Model...")) {
                pendingDialog_ = PendingDialog::ImportModel;
                fileBrowser_.open("Import Model", FileBrowser::Mode::Open,
                                  settings_.modelDir,
                                  {".obj", ".dae", ".gltf", ".glb", ".stl"});
            }

            ImGui::Separator();

            ImGui::MenuItem("Also set as background", nullptr, &environmentAsBackground_);
            if (ImGui::MenuItem("Set Environment...")) {
                pendingDialog_ = PendingDialog::Environment;
                fileBrowser_.open("Set Environment", FileBrowser::Mode::Open,
                                  settings_.environmentDir, {".hdr"});
            }
            if (ImGui::MenuItem("Clear Environment", nullptr, false,
                                document_.scene().environment != nullptr)) {
                clearEnvironment();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit")) pendingAction_ = PendingAction::Quit;

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {

            const auto undoLabel = commands_.canUndo() ? "Undo " + commands_.undoName() : std::string("Undo");
            const auto redoLabel = commands_.canRedo() ? "Redo " + commands_.redoName() : std::string("Redo");

            if (ImGui::MenuItem(undoLabel.c_str(), "Ctrl+Z", false, commands_.canUndo())) {
                commands_.undo();
                document_.setDirty(true);
            }
            if (ImGui::MenuItem(redoLabel.c_str(), "Ctrl+Y", false, commands_.canRedo())) {
                commands_.redo();
                document_.setDirty(true);
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, !selection_.empty())) duplicateSelected();
            if (ImGui::MenuItem("Delete", "Del", false, !selection_.empty())) deleteSelected();
            if (ImGui::MenuItem("Deselect", "Esc", false, !selection_.empty())) selectObject(nullptr);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Add")) {
            drawAddMenu(document_.scene());
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {

            if (ImGui::MenuItem("Frame Selection", "F", false, !selection_.empty())) focusSelected();
            ImGui::Separator();
            if (grid_) ImGui::MenuItem("Grid", nullptr, &grid_->visible);
            if (axes_) ImGui::MenuItem("Origin Axes", nullptr, &axes_->visible);
            ImGui::MenuItem("Bottom Panel", nullptr, &bottomPanelOpen_);
            ImGui::Separator();
            ImGui::MenuItem("Renderer Settings", nullptr, &showRendererSettings);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("Keyboard Shortcuts", nullptr, &showShortcuts);
            ImGui::EndMenu();
        }

        // Right-aligned document name — the same information as the window
        // title, for users running the editor maximised with no title bar in
        // view.
        const auto title = document_.title();
        const float width = ImGui::CalcTextSize(title.c_str()).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - width - ImGui::GetStyle().ItemSpacing.x * 3.f);
        ImGui::TextColored(theme::muted(), "%s", title.c_str());

        ImGui::EndMainMenuBar();
    }

    if (showRendererSettings) {
        // Free-floating (the only such window): it is a transient inspection
        // tool, not part of the fixed layout.
        static RendererSettings settings(*renderer_);
        ImGui::SetNextWindowSize({360 * contentScale_, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowPos({ImGui::GetMainViewport()->Size.x - 380 * contentScale_,
                                 menuHeight_ + toolbarHeight_ + 20 * contentScale_},
                                ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Renderer Settings", &showRendererSettings)) {
            settings.drawContent();
        }
        ImGui::End();
    }

    if (showShortcuts) {
        ImGui::SetNextWindowSize({420 * contentScale_, 0}, ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Keyboard Shortcuts", &showShortcuts,
                         ImGuiWindowFlags_AlwaysAutoResize)) {
            static constexpr const char* rows[][2] = {
                    {"W / E / R", "Translate / rotate / scale gizmo"},
                    {"Q", "Toggle local / world space"},
                    {"Shift (hold)", "Snap while dragging"},
                    {"F", "Frame selection"},
                    {"Del", "Delete selection"},
                    {"Esc", "Deselect"},
                    {"Ctrl+D", "Duplicate"},
                    {"Ctrl+Z", "Undo"},
                    {"Ctrl+Y / Ctrl+Shift+Z", "Redo"},
                    {"Ctrl+N / O / S", "New / open / save scene"},
            };
            if (ImGui::BeginTable("shortcuts", 2, ImGuiTableFlags_SizingStretchProp)) {
                for (const auto& row : rows) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextColored(theme::accent(), "%s", row[0]);
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(row[1]);
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
}
