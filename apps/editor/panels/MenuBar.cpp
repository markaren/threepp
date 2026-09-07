
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../ExampleScenes.hpp"
#include "../ImportFormats.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/imgui/RendererSettings.hpp"

#include "threepp/loaders/AssetSource.hpp"
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

        // Play runs on a snapshot that Stop throws away, so nothing that edits
        // or writes the document is offered while it does — see
        // EditorApp::rejectWhilePlaying(), which is what actually enforces it.
        // Greying the items here is so the reason is visible before the click.
        const bool editable = !isPlaying();

        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("New", "Ctrl+N", false, editable)) pendingAction_ = PendingAction::New;
            if (ImGui::MenuItem("Open...", "Ctrl+O", false, editable)) pendingAction_ = PendingAction::Open;

            if (ImGui::BeginMenu("Open Recent", editable && !settings_.recentFiles().empty())) {
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

            // The scenes that ship inside the binary. Through the same
            // unsaved-changes guard as Open, and to an UNTITLED document — an
            // example is a starting point, not a file you can save over.
            if (ImGui::BeginMenu("Open Example", editable)) {
                for (const auto& example : examples::all()) {
                    if (ImGui::MenuItem(std::string(example.label).c_str())) {
                        if (document_.dirty()) {
                            pendingAction_ = PendingAction::OpenExample;
                            pendingExample_ = std::string(example.slug);
                        } else {
                            openExample(std::string(example.slug));
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", std::string(example.summary).c_str());
                    }
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Save", "Ctrl+S", false, editable)) saveScene();
            if (ImGui::MenuItem("Save As...", nullptr, false, editable)) {
                pendingDialog_ = PendingDialog::SaveAs;
                fileBrowser_.open("Save Scene As", FileBrowser::Mode::Save,
                                  settings_.sceneDir, formats::scenes(),
                                  document_.hasPath() ? document_.path().filename().string()
                                                      : std::string("scene.json"));
            }

            // A prefab is written by the same exporter into the same format —
            // the only difference is which node is the root — so it belongs
            // beside Save As rather than in a category of its own. The scene
            // root is excluded because saving THAT is Save As.
            {
                auto* selected = selection_.get();
                const bool subtree = selected != nullptr && selected != &document_.scene();
                if (ImGui::MenuItem("Save Selection as Prefab...", nullptr, false,
                                    editable && subtree)) {
                    beginSavePrefab(*selected);
                }
            }

            // What Save actually writes. Embedding is the safe default — one
            // file you can hand to anyone — but on a scene with real imported
            // models it is also the reason saving and opening take as long as
            // they do, so the choice is here rather than assumed.
            if (ImGui::BeginMenu("Save Contents")) {

                const auto imageStorage = document_.imageStorage();
                if (ImGui::MenuItem("Embed textures", nullptr, imageStorage == ImageStorage::Embed)) {
                    setImageStorage(ImageStorage::Embed);
                }
                if (ImGui::MenuItem("Reference textures", nullptr, imageStorage == ImageStorage::Reference)) {
                    setImageStorage(ImageStorage::Reference);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Write the path to each texture file instead of a base64 copy.\n"
                                      "Textures with no file of their own (procedural, or packed\n"
                                      "inside a .glb) are still embedded.");
                }

                ImGui::Separator();

                const auto modelStorage = document_.modelStorage();
                if (ImGui::MenuItem("Embed models", nullptr, modelStorage == ModelStorage::Embed)) {
                    setModelStorage(ModelStorage::Embed);
                }
                if (ImGui::MenuItem("Reference models", nullptr, modelStorage == ModelStorage::Reference)) {
                    setModelStorage(ModelStorage::Reference);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Write imported subtrees as a path to the source file plus\n"
                                      "your per-node edits, instead of every vertex. Much smaller\n"
                                      "and much faster, but material edits inside an imported\n"
                                      "subtree are not kept - unlink it first (Edit menu).");
                }

                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Import Model or Robot...")) {
                pendingDialog_ = PendingDialog::ImportModel;
                fileBrowser_.open("Import", FileBrowser::Mode::Open,
                                  settings_.modelDir, formats::importable());
            }

            ImGui::Separator();

            ImGui::MenuItem("Also set as background", nullptr, &environmentAsBackground_);
            if (ImGui::MenuItem("Set Environment...", nullptr, false, editable)) {
                pendingDialog_ = PendingDialog::Environment;
                fileBrowser_.open("Set Environment", FileBrowser::Mode::Open,
                                  settings_.environmentDir, formats::environments());
            }
            if (ImGui::MenuItem("Clear Environment", nullptr, false,
                                editable && document_.scene().environment != nullptr)) {
                clearEnvironment();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit")) pendingAction_ = PendingAction::Quit;

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {

            const auto undoLabel = commands_.canUndo() ? "Undo " + commands_.undoName() : std::string("Undo");
            const auto redoLabel = commands_.canRedo() ? "Redo " + commands_.redoName() : std::string("Redo");

            if (ImGui::MenuItem(undoLabel.c_str(), "Ctrl+Z", false, editable && commands_.canUndo())) undo();
            if (ImGui::MenuItem(redoLabel.c_str(), "Ctrl+Y", false, editable && commands_.canRedo())) redo();

            ImGui::Separator();

            // Breaking the link makes the subtree ordinary scene content, so a
            // save writes it out in full. That is what you want before editing
            // its materials, or when the source file is about to go away.
            {
                auto* selected = selection_.get();
                const bool linked = selected && !assetSource(*selected).empty();
                if (ImGui::MenuItem("Unlink Imported Asset", nullptr, false, editable && linked)) {
                    unlinkSelectedAsset();
                }
                if (linked && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Linked to %s", assetSource(*selected).filename().string().c_str());
                }
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, editable && !selection_.empty())) duplicateSelected();
            if (ImGui::MenuItem("Delete", "Del", false, editable && !selection_.empty())) deleteSelected();
            // Deselect stays: selection is editor state, not document state.
            if (ImGui::MenuItem("Deselect", "Esc", false, !selection_.empty())) selectObject(nullptr);

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Add", editable)) {
            drawAddMenu(document_.scene());
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {

            if (ImGui::MenuItem("Frame Selection", "F", false, !selection_.empty())) focusSelected();
            // Frame Selection, but every frame. Enabled with nothing selected:
            // it is armed then, and the next click starts the chase.
            {
                bool follow = followSelection();
                if (ImGui::MenuItem("Follow Selection", "Shift+F", &follow)) setFollowSelection(follow);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Keep the selection in view: the orbit target tracks it and the\n"
                                      "camera rides along, so your angle and distance are kept and you\n"
                                      "can still orbit and zoom. Works while playing - that is the point.");
                }
            }
            ImGui::Separator();

            if (ImGui::BeginMenu("Viewpoint")) {

                // The numpad layout every 3D editor shares. Picking one of the
                // six switches to the orthographic projection, which is what an
                // axis view is for; the toggle below is how to come back.
                static constexpr struct {
                    ViewPreset preset;
                    const char* shortcut;
                } viewpoints[] = {
                        {ViewPreset::Front, "Num1 / Alt+1"},
                        {ViewPreset::Back, "Ctrl+Num1"},
                        {ViewPreset::Right, "Num3 / Alt+3"},
                        {ViewPreset::Left, "Ctrl+Num3"},
                        {ViewPreset::Top, "Num7 / Alt+7"},
                        {ViewPreset::Bottom, "Ctrl+Num7"},
                };
                for (const auto& entry : viewpoints) {
                    if (ImGui::MenuItem(viewPresetLabel(entry.preset), entry.shortcut,
                                        viewPreset() == entry.preset && orthographic())) {
                        setOrthographic(true);
                        setViewPreset(entry.preset);
                    }
                }
                ImGui::Separator();
                bool ortho = orthographic();
                if (ImGui::MenuItem("Orthographic", "Num5 / Alt+5", &ortho)) setOrthographic(ortho);

                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (grid_) ImGui::MenuItem("Grid", nullptr, &grid_->visible);
            if (axes_) ImGui::MenuItem("Origin Axes", nullptr, &axes_->visible);
            // Only draws anything while playing — the colliders and joints do
            // not exist before that — but the toggle is a preference, not a
            // play state.
            ImGui::MenuItem("Physics Debug", nullptr, &physicsDebug_);
            // On by default, unlike the colliders: a sensor's cloud IS the
            // sensor as far as the viewport is concerned, and a feature nobody
            // switches on is a feature nobody knows works. Costs nothing in a
            // scene with no sensors authored.
            ImGui::MenuItem("Sensor Point Cloud", nullptr, &sensorCloudVisible_);
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
                                 menuHeight_ + 20 * contentScale_},
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
                    {"Shift+F", "Follow selection (chase camera, works while playing)"},
                    {"Num1 / 3 / 7", "Front / right / top orthographic view"},
                    {"Ctrl+Num1 / 3 / 7", "Back / left / bottom"},
                    {"Num5", "Toggle orthographic / perspective"},
                    {"Alt+1 / 3 / 5 / 7", "The same, without a numpad"},
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
