
#include "threepp/extras/editor/EditorSettings.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::string environmentVariable(const char* name) {

        const char* value = std::getenv(name);
        return value ? std::string(value) : std::string{};
    }

    std::string getString(const nlohmann::json& j, const char* key) {

        if (!j.contains(key) || !j[key].is_string()) return {};
        return j[key].get<std::string>();
    }

    // Stored as words, not enum ordinals: the file is meant to be readable and
    // hand-editable, and a reordered enum must not silently mean something
    // else the next time it is read.
    const char* storageName(ImageStorage storage) {

        switch (storage) {
            case ImageStorage::Reference: return "reference";
            case ImageStorage::Omit: return "omit";
            default: return "embed";
        }
    }

    ImageStorage imageStorageFrom(const std::string& text, ImageStorage fallback) {

        if (text == "reference") return ImageStorage::Reference;
        if (text == "embed") return ImageStorage::Embed;
        // "omit" is deliberately not accepted: it produces a document that
        // cannot render on its own, which is a play-snapshot concern and never
        // something a user should be able to pick for File ▸ Save.
        return fallback;
    }

    ModelStorage modelStorageFrom(const std::string& text, ModelStorage fallback) {

        if (text == "reference") return ModelStorage::Reference;
        if (text == "embed") return ModelStorage::Embed;
        return fallback;
    }

}// namespace


std::filesystem::path EditorSettings::defaultPath() {

    std::filesystem::path base;

#ifdef _WIN32
    if (auto appData = environmentVariable("APPDATA"); !appData.empty()) {
        base = appData;
    }
#else
    if (auto xdg = environmentVariable("XDG_CONFIG_HOME"); !xdg.empty()) {
        base = xdg;
    } else if (auto home = environmentVariable("HOME"); !home.empty()) {
        base = std::filesystem::path(home) / ".config";
    }
#endif

    // No home directory to speak of (a service account, a stripped container):
    // fall back to the working directory rather than refusing to remember
    // anything.
    if (base.empty()) base = std::filesystem::current_path();

    return base / "threepp-editor" / "settings.json";
}

bool EditorSettings::load(const std::filesystem::path& path) {

    std::ifstream file(path);
    if (!file.is_open()) return false;

    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception&) {
        return false;
    }
    if (!j.is_object()) return false;

    sceneDir = getString(j, "sceneDir");
    modelDir = getString(j, "modelDir");
    textureDir = getString(j, "textureDir");
    environmentDir = getString(j, "environmentDir");
    scriptDir = getString(j, "scriptDir");
    soundDir = getString(j, "soundDir");

    if (j.contains("bottomPanelOpen") && j["bottomPanelOpen"].is_boolean()) {
        bottomPanelOpen = j["bottomPanelOpen"].get<bool>();
    }

    imageStorage = imageStorageFrom(getString(j, "imageStorage"), imageStorage);
    modelStorage = modelStorageFrom(getString(j, "modelStorage"), modelStorage);

    // Clamped on read: a settings file edited by hand (or written by a build
    // with different limits) must not be able to push a panel off-screen.
    const auto readSize = [&j](const char* key, float fallback, float lo, float hi) {
        if (!j.contains(key) || !j[key].is_number()) return fallback;
        return std::clamp(j[key].get<float>(), lo, hi);
    };
    hierarchyWidth = readSize("hierarchyWidth", hierarchyWidth, minPanelWidth, maxPanelWidth);
    inspectorWidth = readSize("inspectorWidth", inspectorWidth, minPanelWidth, maxPanelWidth);
    bottomPanelHeight = readSize("bottomPanelHeight", bottomPanelHeight,
                                 minBottomHeight, maxBottomHeight);

    recentFiles_.clear();
    if (j.contains("recentFiles") && j["recentFiles"].is_array()) {
        for (const auto& entry : j["recentFiles"]) {
            if (!entry.is_string()) continue;
            auto value = entry.get<std::string>();
            // A recent entry pointing at a deleted file is noise in the menu.
            std::error_code ec;
            if (!std::filesystem::exists(value, ec)) continue;
            recentFiles_.push_back(std::move(value));
            if (recentFiles_.size() >= maxRecentFiles) break;
        }
    }

    return true;
}

bool EditorSettings::save(const std::filesystem::path& path) const {

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    nlohmann::json j;
    j["sceneDir"] = sceneDir;
    j["modelDir"] = modelDir;
    j["textureDir"] = textureDir;
    j["environmentDir"] = environmentDir;
    j["scriptDir"] = scriptDir;
    j["soundDir"] = soundDir;
    j["bottomPanelOpen"] = bottomPanelOpen;
    j["imageStorage"] = storageName(imageStorage);
    j["modelStorage"] = modelStorage == ModelStorage::Reference ? "reference" : "embed";
    j["hierarchyWidth"] = hierarchyWidth;
    j["inspectorWidth"] = inspectorWidth;
    j["bottomPanelHeight"] = bottomPanelHeight;
    j["recentFiles"] = recentFiles_;

    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << j.dump(2) << "\n";
    return file.good();
}

void EditorSettings::addRecentFile(const std::filesystem::path& path) {

    if (path.empty()) return;

    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(path, ec);
    const auto value = (ec ? path : canonical).string();

    recentFiles_.erase(std::remove(recentFiles_.begin(), recentFiles_.end(), value), recentFiles_.end());
    recentFiles_.insert(recentFiles_.begin(), value);
    if (recentFiles_.size() > maxRecentFiles) recentFiles_.resize(maxRecentFiles);
}
