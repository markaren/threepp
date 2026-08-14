
#include "threepp/extras/editor/SceneDocument.hpp"

#include "threepp/extras/editor/SceneSnapshot.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <fstream>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // RAII around the editor-only detach, so an exception thrown mid-export can
    // never leave the viewport without its grid and gizmo.
    struct ExportScope {

        explicit ExportScope(std::function<void()> attach)
            : attach_(std::move(attach)) {}

        ~ExportScope() { attach_(); }

        std::function<void()> attach_;
    };

    // A document whose root is a Group or a Mesh is perfectly legal three.js
    // JSON (ObjectExporter writes whatever root it is given). Adopt it as scene
    // content rather than refusing to open it. Shared by both parse paths so
    // "opened from a file" and "opened from text" cannot drift apart.
    std::shared_ptr<Scene> asScene(std::shared_ptr<Object3D> parsed) {

        if (auto scene = std::dynamic_pointer_cast<Scene>(parsed)) return scene;

        auto scene = Scene::create();
        scene->name = "Scene";
        scene->add(std::move(parsed));
        return scene;
    }

}// namespace


SceneDocument::SceneDocument()
    : scene_(Scene::create()) {

    scene_->name = "Scene";
}

std::string SceneDocument::displayName() const {

    return path_.empty() ? std::string("untitled") : path_.filename().string();
}

std::string SceneDocument::title() const {

    return displayName() + (dirty_ ? "*" : "");
}

void SceneDocument::newScene() {

    auto scene = Scene::create();
    scene->name = "Scene";
    path_.clear();
    warnings_.clear();
    replaceScene(std::move(scene));
    dirty_ = false;
}

std::shared_ptr<Object3D> SceneDocument::runLoader(
        const std::function<std::shared_ptr<Object3D>(ObjectLoader&)>& load,
        const std::string& describe, std::string* error) {

    ObjectLoader loader;
    std::shared_ptr<Object3D> parsed;
    try {
        parsed = load(loader);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }

    warnings_ = loader.warnings();

    if (!parsed && error) *error = "could not parse " + describe;
    return parsed;
}

bool SceneDocument::open(const std::filesystem::path& path, std::string* error) {

    warnings_.clear();

    if (!std::filesystem::exists(path)) {
        if (error) *error = "file not found: " + path.string();
        return false;
    }

    auto parsed = runLoader([&](ObjectLoader& l) { return l.load(path); }, path.string(), error);
    if (!parsed) return false;

    path_ = path;
    replaceScene(asScene(std::move(parsed)));
    dirty_ = false;
    return true;
}

bool SceneDocument::openJson(const std::string& jsonText, std::string* error) {

    warnings_.clear();

    if (jsonText.empty()) {
        if (error) *error = "empty document";
        return false;
    }

    auto parsed = runLoader([&](ObjectLoader& l) { return l.parse(jsonText); }, "the document", error);
    if (!parsed) return false;

    // No path: this document came from nowhere on disk, and Save has to ask.
    path_.clear();
    replaceScene(asScene(std::move(parsed)));
    dirty_ = false;
    return true;
}

bool SceneDocument::save(std::string* error) {

    if (path_.empty()) {
        if (error) *error = "no path set — use Save As";
        return false;
    }
    return saveAs(path_, error);
}

bool SceneDocument::saveAs(const std::filesystem::path& path, std::string* error) {

    warnings_.clear();

    detachEditorOnly();
    ExportScope scope([this] { attachEditorOnly(); });

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.images = imageStorage_;
    options.models = modelStorage_;
    options.prettyPrint = true;

    try {
        // save() derives the base for relative references from `path`.
        exporter.save(*scene_, path, options);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        warnings_ = exporter.warnings();
        return false;
    }

    warnings_ = exporter.warnings();
    path_ = path;
    dirty_ = false;
    return true;
}

bool SceneDocument::exportSubtree(Object3D& object, const std::filesystem::path& path, std::string* error) {

    warnings_.clear();

    // The subtree is usually nowhere near the overlays, but "usually" is not an
    // invariant: a helper registered under a scene object would ride along into
    // the prefab. Detached for exactly the reason saveAs() detaches them.
    detachEditorOnly();
    ExportScope scope([this] { attachEditorOnly(); });

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.images = imageStorage_;
    options.models = modelStorage_;
    // Auto, so a prefab named ".tpz" is an archive on the same terms a document
    // is — a prefab IS a document, and nothing here gets to decide otherwise.
    options.format = DocumentFormat::Auto;
    options.prettyPrint = true;

    try {
        exporter.save(object, path, options);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        warnings_ = exporter.warnings();
        return false;
    }

    warnings_ = exporter.warnings();
    return true;
}

std::string SceneDocument::toJson(bool prettyPrint, std::string* error) {

    warnings_.clear();

    detachEditorOnly();
    ExportScope scope([this] { attachEditorOnly(); });

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.images = imageStorage_;
    options.models = modelStorage_;
    // No file to make references relative to, so this one uses the document's
    // own directory when it has been saved before, and absolute paths if not.
    options.resourcePath = path_.empty() ? std::filesystem::path{} : path_.parent_path();
    options.prettyPrint = prettyPrint;

    try {
        auto json = exporter.toJson(*scene_, options);
        warnings_ = exporter.warnings();
        return json;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return {};
    }
}

void SceneDocument::replaceScene(std::shared_ptr<Scene> scene) {

    if (!scene) return;

    scene_ = std::move(scene);
    // Helpers follow the document, not the old scene object.
    attachEditorOnly();

    const auto snapshot = listeners_;
    for (const auto& listener : snapshot) listener.fn(*scene_);
}

int SceneDocument::onSceneReplaced(std::function<void(Scene&)> listener) {

    if (!listener) return 0;
    const int id = nextListenerId_++;
    listeners_.push_back({id, std::move(listener)});
    return id;
}

void SceneDocument::removeListener(int id) {

    listeners_.erase(std::remove_if(listeners_.begin(), listeners_.end(),
                                    [id](const Listener& l) { return l.id == id; }),
                     listeners_.end());
}

void SceneDocument::addEditorOnly(Object3D& object) {

    if (isEditorOnly(object)) return;
    editorOnly_.push_back(&object);
    if (object.parent != scene_.get()) scene_->addRef(object);
}

void SceneDocument::removeEditorOnly(Object3D& object) {

    editorOnly_.erase(std::remove(editorOnly_.begin(), editorOnly_.end(), &object), editorOnly_.end());
    if (object.parent == scene_.get()) object.removeFromParent();
}

bool SceneDocument::isEditorOnly(const Object3D& object) const {

    // Direct membership is not enough: the selection outline and the gizmo's
    // own handles are children of a registered overlay node.
    for (const Object3D* o = &object; o != nullptr; o = o->parent) {
        if (std::find(editorOnly_.begin(), editorOnly_.end(), o) != editorOnly_.end()) return true;
    }
    return false;
}

bool SceneDocument::capture(SceneSnapshot& snapshot, std::string* error) {

    detachEditorOnly();
    ExportScope scope([this] { attachEditorOnly(); });

    return snapshot.capture(*scene_, error);
}

bool SceneDocument::restore(const SceneSnapshot& snapshot, std::string* error) {

    auto scene = snapshot.restore(error);
    if (!scene) return false;

    replaceScene(std::move(scene));
    return true;
}

void SceneDocument::detachEditorOnly() {

    for (auto* object : editorOnly_) {
        if (object->parent == scene_.get()) object->removeFromParent();
    }
}

void SceneDocument::attachEditorOnly() {

    for (auto* object : editorOnly_) {
        if (object->parent != scene_.get()) scene_->addRef(*object);
    }
}
