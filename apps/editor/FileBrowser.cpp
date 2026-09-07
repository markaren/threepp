
#include "FileBrowser.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstring>

using namespace threepp::editor;

namespace {

    std::string toLower(std::string text) {

        std::transform(text.begin(), text.end(), text.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return text;
    }

    void copyInto(char* buffer, std::size_t size, const std::string& text) {

        const auto n = std::min(text.size(), size - 1);
        std::memcpy(buffer, text.data(), n);
        buffer[n] = '\0';
    }

}// namespace


void FileBrowser::open(std::string title,
                       Mode mode,
                       const std::filesystem::path& startDirectory,
                       std::vector<std::string> extensions,
                       const std::string& defaultName) {

    title_ = std::move(title);
    mode_ = mode;
    extensions_.clear();
    for (auto& extension : extensions) extensions_.push_back(toLower(extension));

    std::error_code ec;
    auto directory = startDirectory;
    if (directory.empty() || !std::filesystem::is_directory(directory, ec)) {
        directory = std::filesystem::current_path(ec);
    }
    setDirectory(directory);

    copyInto(nameBuffer_, sizeof(nameBuffer_), defaultName);
    result_.clear();
    error_.clear();
    selected_ = -1;
    open_ = true;
    justOpened_ = true;
    confirmed_ = false;
}

void FileBrowser::close() {

    open_ = false;
}

void FileBrowser::setDirectory(const std::filesystem::path& directory) {

    std::error_code ec;
    auto resolved = std::filesystem::weakly_canonical(directory, ec);
    directory_ = ec ? directory : resolved;
    copyInto(pathBuffer_, sizeof(pathBuffer_), directory_.string());
    selected_ = -1;
    refresh();
}

void FileBrowser::refresh() {

    entries_.clear();
    error_.clear();

    std::error_code ec;
    std::filesystem::directory_iterator it(directory_, ec);
    if (ec) {
        error_ = "cannot list " + directory_.string();
        return;
    }

    for (const auto& entry : it) {
        std::error_code entryEc;
        const bool isDirectory = entry.is_directory(entryEc);
        if (entryEc) continue;
        if (!isDirectory && !matches(entry.path())) continue;
        entries_.push_back({entry.path().filename().string(), isDirectory});
    }

    // Directories first, then files, each alphabetically — the ordering every
    // file dialog uses, and the one users scan fastest.
    std::sort(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
        if (a.isDirectory != b.isDirectory) return a.isDirectory;
        return toLower(a.name) < toLower(b.name);
    });
}

bool FileBrowser::matches(const std::filesystem::path& path) const {

    if (extensions_.empty()) return true;
    const auto extension = toLower(path.extension().string());
    return std::find(extensions_.begin(), extensions_.end(), extension) != extensions_.end();
}

void FileBrowser::confirm(const std::filesystem::path& path) {

    result_ = path;
    confirmed_ = true;
    open_ = false;
}

bool FileBrowser::draw(float scale) {

    if (!open_) return false;

    confirmed_ = false;

    if (justOpened_) {
        ImGui::OpenPopup(title_.c_str());
        justOpened_ = false;
    }

    const ImVec2 centre = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(centre, ImGuiCond_Appearing, {0.5f, 0.5f});
    ImGui::SetNextWindowSize({640 * scale, 460 * scale}, ImGuiCond_Appearing);

    bool stayOpen = true;
    if (!ImGui::BeginPopupModal(title_.c_str(), &stayOpen, ImGuiWindowFlags_NoSavedSettings)) {
        // The popup was dismissed (Esc / title-bar close).
        if (!ImGui::IsPopupOpen(title_.c_str())) open_ = false;
        return false;
    }

    // --- location bar -------------------------------------------------------
    if (ImGui::Button("Up")) {
        if (directory_.has_parent_path() && directory_.parent_path() != directory_) {
            setDirectory(directory_.parent_path());
        }
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-70 * scale);
    if (ImGui::InputText("##path", pathBuffer_, sizeof(pathBuffer_),
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        setDirectory(std::filesystem::path(pathBuffer_));
    }
    ImGui::SameLine();
    if (ImGui::Button("Go")) setDirectory(std::filesystem::path(pathBuffer_));

    if (!error_.empty()) {
        ImGui::TextColored({0.9f, 0.4f, 0.4f, 1.f}, "%s", error_.c_str());
    }

    // --- listing ------------------------------------------------------------
    const float footer = ImGui::GetFrameHeightWithSpacing() * 2.f + ImGui::GetStyle().ItemSpacing.y;
    if (ImGui::BeginChild("##entries", {0, -footer}, ImGuiChildFlags_Borders)) {

        for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
            const auto& entry = entries_[i];
            const std::string label = (entry.isDirectory ? "[ ] " : "    ") + entry.name;
            const bool selected = selected_ == i;
            if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
                selected_ = i;
                if (!entry.isDirectory) copyInto(nameBuffer_, sizeof(nameBuffer_), entry.name);
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    if (entry.isDirectory) {
                        setDirectory(directory_ / entry.name);
                        break;
                    }
                    confirm(directory_ / entry.name);
                    ImGui::CloseCurrentPopup();
                    ImGui::EndChild();
                    ImGui::EndPopup();
                    return true;
                }
            }
        }
    }
    ImGui::EndChild();

    // --- filename + actions -------------------------------------------------
    ImGui::SetNextItemWidth(-1);
    const bool submitted = ImGui::InputText("##filename", nameBuffer_, sizeof(nameBuffer_),
                                            ImGuiInputTextFlags_EnterReturnsTrue);

    const bool hasName = nameBuffer_[0] != '\0';
    if (!hasName) ImGui::BeginDisabled();
    const bool accept = ImGui::Button(mode_ == Mode::Save ? "Save" : "Open", {110 * scale, 0}) || submitted;
    if (!hasName) ImGui::EndDisabled();

    ImGui::SameLine();
    if (ImGui::Button("Cancel", {110 * scale, 0})) {
        open_ = false;
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        return false;
    }

    if (accept && hasName) {
        auto chosen = directory_ / nameBuffer_;
        std::error_code ec;
        if (std::filesystem::is_directory(chosen, ec)) {
            setDirectory(chosen);
        } else {
            // Save with no extension typed: adopt the filter's first one, so
            // "myscene" becomes "myscene.json" rather than an extensionless file
            // the editor cannot reopen.
            if (mode_ == Mode::Save && !extensions_.empty() && !chosen.has_extension()) {
                chosen += extensions_.front();
            }
            confirm(chosen);
            ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
            return true;
        }
    }

    ImGui::EndPopup();

    if (!stayOpen) open_ = false;
    return false;
}
