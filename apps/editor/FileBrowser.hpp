// A minimal, first-party file dialog drawn with ImGui.
//
// The editor deliberately does not pull in a native-dialog library: one modal
// popup over std::filesystem covers open and save, filters by extension, and
// behaves identically on every platform threepp builds for.
//
// Usage:
//   browser.open("Open Scene", Mode::Open, startDir, {".json"});
//   if (browser.draw(scale)) { use(browser.result()); }

#ifndef THREEPP_EDITOR_FILEBROWSER_HPP
#define THREEPP_EDITOR_FILEBROWSER_HPP

#include <filesystem>
#include <string>
#include <vector>

namespace threepp::editor {

    class FileBrowser {

    public:
        enum class Mode {
            Open,
            Save
        };

        // `extensions` are lower-case and include the dot (".json"). Empty means
        // "show every file".
        void open(std::string title,
                  Mode mode,
                  const std::filesystem::path& startDirectory,
                  std::vector<std::string> extensions,
                  const std::string& defaultName = {});

        // Draws the modal when open. Returns true exactly once, on the frame the
        // user confirmed a path.
        bool draw(float scale);

        [[nodiscard]] bool isOpen() const { return open_; }

        // Valid on the frame draw() returned true.
        [[nodiscard]] const std::filesystem::path& result() const { return result_; }

        // The directory the browser is showing — worth persisting so the next
        // dialog of the same kind starts where the user left off.
        [[nodiscard]] const std::filesystem::path& directory() const { return directory_; }

        void close();

    private:
        struct Entry {
            std::string name;
            bool isDirectory;
        };

        void setDirectory(const std::filesystem::path& directory);
        void refresh();
        [[nodiscard]] bool matches(const std::filesystem::path& path) const;
        void confirm(const std::filesystem::path& path);

        bool open_ = false;
        bool justOpened_ = false;
        bool confirmed_ = false;
        Mode mode_ = Mode::Open;
        std::string title_ = "Select File";
        std::filesystem::path directory_;
        std::filesystem::path result_;
        std::vector<std::string> extensions_;
        std::vector<Entry> entries_;
        std::string error_;
        int selected_ = -1;
        char nameBuffer_[512]{};
        char pathBuffer_[1024]{};
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_FILEBROWSER_HPP
