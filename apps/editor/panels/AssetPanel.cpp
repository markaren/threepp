
#include "../EditorApp.hpp"
#include "../ImportFormats.hpp"
#include "../EditorTheme.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <algorithm>
#include <system_error>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    enum class AssetKind {
        Scene,
        Model,
        Environment,
        Image,
        Other
    };

    AssetKind classify(const std::filesystem::path& path) {

        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (extension == ".json") return AssetKind::Scene;
        if (formats::contains(formats::importable(), extension)) return AssetKind::Model;
        if (extension == ".hdr") return AssetKind::Environment;
        if (formats::isImage(extension)) return AssetKind::Image;
        return AssetKind::Other;
    }

    const char* kindLabel(AssetKind kind) {

        switch (kind) {
            case AssetKind::Scene: return "scene";
            case AssetKind::Model: return "model";
            case AssetKind::Environment: return "hdr";
            case AssetKind::Image: return "image";
            case AssetKind::Other: return "";
        }
        return "";
    }

}// namespace


void EditorApp::drawAssetsTab() {

    const float s = contentScale_;

    if (ImGui::Button("Up")) {
        if (assetDir_.has_parent_path() && assetDir_.parent_path() != assetDir_) {
            assetDir_ = assetDir_.parent_path();
        }
    }
    ImGui::SameLine();
    ImGui::TextColored(theme::muted(), "%s", assetDir_.string().c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Scene folder") && document_.hasPath()) {
        assetDir_ = document_.path().parent_path();
    }

    ImGui::Separator();

    if (!ImGui::BeginChild("##assets", {0, 0})) {
        ImGui::EndChild();
        return;
    }

    std::error_code ec;
    std::filesystem::directory_iterator it(assetDir_, ec);
    if (ec) {
        ImGui::TextColored(theme::danger(), "cannot list %s", assetDir_.string().c_str());
        ImGui::EndChild();
        return;
    }

    std::vector<std::filesystem::path> directories;
    std::vector<std::filesystem::path> files;
    for (const auto& entry : it) {
        std::error_code entryEc;
        if (entry.is_directory(entryEc)) {
            directories.push_back(entry.path());
        } else if (classify(entry.path()) != AssetKind::Other) {
            files.push_back(entry.path());
        }
    }
    std::sort(directories.begin(), directories.end());
    std::sort(files.begin(), files.end());

    // Two columns of names — a flat grid reads faster than a single tall list
    // in a 200 px strip.
    const int columns = std::max(1, static_cast<int>(ImGui::GetContentRegionAvail().x / (220 * s)));
    if (!ImGui::BeginTable("##assetGrid", columns, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::EndChild();
        return;
    }

    for (const auto& directory : directories) {
        ImGui::TableNextColumn();
        ImGui::PushID(directory.string().c_str());
        if (ImGui::Selectable(("[dir] " + directory.filename().string()).c_str(), false,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                // Deferred: changing the directory invalidates the iterator we
                // are still walking.
                auto next = directory;
                deferred_ = [this, next] { assetDir_ = next; };
            }
        }
        ImGui::PopID();
    }

    for (const auto& file : files) {
        const auto kind = classify(file);
        ImGui::TableNextColumn();
        ImGui::PushID(file.string().c_str());
        if (ImGui::Selectable(file.filename().string().c_str(), false,
                              ImGuiSelectableFlags_AllowDoubleClick)) {
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                auto target = file;
                switch (kind) {
                    case AssetKind::Scene:
                        deferred_ = [this, target] {
                            if (document_.dirty()) {
                                pendingAction_ = PendingAction::OpenPath;
                                pendingPath_ = target;
                            } else {
                                openScene(target);
                            }
                        };
                        break;
                    case AssetKind::Model:
                        deferred_ = [this, target] { importModel(target); };
                        break;
                    case AssetKind::Environment:
                        deferred_ = [this, target] { setEnvironment(target, environmentAsBackground_); };
                        break;
                    case AssetKind::Image:
                        deferred_ = [this, target] { assignTextureToSelection(target); };
                        break;
                    case AssetKind::Other:
                        break;
                }
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s\ndouble-click to %s", file.string().c_str(),
                              kind == AssetKind::Scene ? "open" : kind == AssetKind::Model ? "import"
                                                             : kind == AssetKind::Environment
                                                                     ? "set as environment"
                                                                     : "assign as base colour map");
        }
        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "%s", kindLabel(kind));
        ImGui::PopID();
    }

    ImGui::EndTable();
    ImGui::EndChild();
}

void EditorApp::drawConsoleTab() {

    if (ImGui::SmallButton("Clear")) console_.clear();
    ImGui::SameLine();
    ImGui::TextColored(theme::muted(), "%zu messages", console_.size());
    ImGui::Separator();

    if (ImGui::BeginChild("##console", {0, 0})) {
        for (const auto& line : console_) {
            const bool bad = line.rfind("warning", 0) == 0 || line.find("failed") != std::string::npos;
            if (bad) {
                ImGui::TextColored(theme::warning(), "%s", line.c_str());
            } else {
                ImGui::TextUnformatted(line.c_str());
            }
        }
        if (consoleScrollToBottom_) {
            ImGui::SetScrollHereY(1.f);
            consoleScrollToBottom_ = false;
        }
    }
    ImGui::EndChild();
}

void EditorApp::drawBottomPanel() {

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    const float left = layout::hierarchyWidth * s;
    const float right = layout::inspectorWidth * s;
    const float collapsedHeight = ImGui::GetFrameHeight() + 6 * s;
    const float height = bottomPanelOpen_ ? layout::bottomHeight * s : collapsedHeight;

    ImGui::SetNextWindowPos({viewport->Pos.x + left,
                             viewport->Pos.y + viewport->Size.y - statusHeight_ - height});
    ImGui::SetNextWindowSize({std::max(viewport->Size.x - left - right, 120.f * s), height});

    if (ImGui::Begin("##bottom", nullptr, layout::barFlags)) {

        if (ImGui::SmallButton(bottomPanelOpen_ ? "v" : "^")) bottomPanelOpen_ = !bottomPanelOpen_;
        ImGui::SameLine();

        if (!bottomPanelOpen_) {
            ImGui::TextColored(theme::muted(), "Assets / Console");
            if (!console_.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(theme::muted(), "- %s", console_.back().c_str());
            }
        } else if (ImGui::BeginTabBar("##bottomTabs")) {
            if (ImGui::BeginTabItem("Assets")) {
                drawAssetsTab();
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Console")) {
                drawConsoleTab();
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}
