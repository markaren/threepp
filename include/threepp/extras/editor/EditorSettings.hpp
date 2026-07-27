// Small persistent preferences file for the editor: recent scenes and the
// directories the file browser should open in.
//
// Lives in the library (not the app) so it can use the vendored nlohmann JSON
// the rest of threepp already serializes with, rather than inventing a second
// format or dragging a parser into the executable.
//
// Location: %APPDATA%/threepp-editor/settings.json on Windows,
// $XDG_CONFIG_HOME (or ~/.config)/threepp-editor/settings.json elsewhere.
// A missing or unreadable file is not an error — the editor just starts with
// defaults, which is the only sensible behaviour for preferences.

#ifndef THREEPP_EDITOR_EDITORSETTINGS_HPP
#define THREEPP_EDITOR_EDITORSETTINGS_HPP

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

        // Where the file browser starts for each kind of file.
        std::string sceneDir;
        std::string modelDir;
        std::string textureDir;
        std::string environmentDir;
        std::string scriptDir;

        bool bottomPanelOpen = true;

        // Side panel widths, in pixels at 100% DPI (the content scale is
        // applied at draw time). User-draggable, so a deep hierarchy or a long
        // material name is a resize away rather than permanently clipped.
        float hierarchyWidth = 280.f;
        float inspectorWidth = 340.f;

        static constexpr float minPanelWidth = 180.f;
        static constexpr float maxPanelWidth = 720.f;

    private:
        std::vector<std::string> recentFiles_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_EDITORSETTINGS_HPP
