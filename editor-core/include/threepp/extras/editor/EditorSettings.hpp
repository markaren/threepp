// Small persistent preferences file for the editor: recent scenes and the
// directories the file browser should open in.
//
// Lives in editor-core (not the app) so it can use the vendored nlohmann JSON
// the rest of threepp already serializes with, rather than inventing a second
// format or dragging a parser into the executable.
//
// Location: %APPDATA%/threepp-editor/settings.json on Windows,
// $XDG_CONFIG_HOME (or ~/.config)/threepp-editor/settings.json elsewhere.
// A missing or unreadable file is not an error — the editor just starts with
// defaults, which is the only sensible behaviour for preferences.

#ifndef THREEPP_EDITOR_EDITORSETTINGS_HPP
#define THREEPP_EDITOR_EDITORSETTINGS_HPP

#include "threepp/loaders/ObjectExporter.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace threepp::editor {

    class EditorSettings {

    public:
        static constexpr std::size_t maxRecentFiles = 10;

        [[nodiscard]] static std::filesystem::path defaultPath();

        bool load(const std::filesystem::path& path);
        bool save(const std::filesystem::path& path) const;

        // Most recent first, de-duplicated, capped at maxRecentFiles. Entries
        // that no longer exist on disk are dropped on load.
        [[nodiscard]] const std::vector<std::string>& recentFiles() const { return recentFiles_; }
        void addRecentFile(const std::filesystem::path& path);
        void clearRecentFiles() { recentFiles_.clear(); }

        // Where the file browser starts for each kind of file. `prefabDir` is
        // its own entry rather than a reuse of sceneDir because a prefab library
        // is somewhere else than the scene you are building with it.
        std::string sceneDir;
        std::string prefabDir;
        std::string modelDir;
        std::string textureDir;
        std::string environmentDir;
        std::string scriptDir;
        std::string soundDir;

        bool bottomPanelOpen = true;

        // How File ▸ Save writes textures and imported models: embedded (the
        // document stands alone) or referenced (the document points at the
        // source files, and saves and opens in a fraction of the time).
        //
        // A preference rather than a document property — it describes how you
        // work, so a scene saved by reference on one machine still opens on
        // another that prefers to embed.
        ImageStorage imageStorage = ImageStorage::Embed;
        ModelStorage modelStorage = ModelStorage::Embed;

        // Side panel widths, in pixels at 100% DPI (the content scale is
        // applied at draw time). User-draggable, so a deep hierarchy or a long
        // material name is a resize away rather than permanently clipped.
        float hierarchyWidth = 280.f;
        float inspectorWidth = 340.f;

        static constexpr float minPanelWidth = 180.f;
        static constexpr float maxPanelWidth = 720.f;

        // Bottom panel height, same units and draggable for the same reason:
        // the console is fine in a 200 px strip and the docked script editor is
        // not. The viewport is the real upper bound — see
        // EditorApp::bottomHeightLimit() — so this cap only guards the file.
        float bottomPanelHeight = 200.f;

        static constexpr float minBottomHeight = 90.f;
        static constexpr float maxBottomHeight = 1200.f;

    private:
        std::vector<std::string> recentFiles_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_EDITORSETTINGS_HPP
