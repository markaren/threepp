// The thing the editor has open: one Scene, where it came from, and whether it
// has unsaved changes.
//
// Save/load go through ObjectExporter / ObjectLoader — three.js "Object" JSON,
// version 4.5. There is no editor-specific sidecar format; anything the editor
// can set is either already in that schema or rides in userData (see
// PhysicsConfig), which serializes with it.
//
// Editor-only objects (grid, gizmo, selection outline) have to be in the scene
// to be rendered but must never reach the file. Register them with
// addEditorOnly() and the document detaches them around every export.

#ifndef THREEPP_EDITOR_SCENEDOCUMENT_HPP
#define THREEPP_EDITOR_SCENEDOCUMENT_HPP

#include "threepp/loaders/ObjectExporter.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;
    class Scene;

}// namespace threepp

namespace threepp::editor {

    class SceneSnapshot;

    class SceneDocument {

    public:
        SceneDocument();

        [[nodiscard]] Scene& scene() const { return *scene_; }
        [[nodiscard]] std::shared_ptr<Scene> scenePtr() const { return scene_; }

        [[nodiscard]] const std::filesystem::path& path() const { return path_; }
        [[nodiscard]] bool hasPath() const { return !path_.empty(); }

        [[nodiscard]] bool dirty() const { return dirty_; }
        void setDirty(bool dirty) { dirty_ = dirty; }

        // "untitled" / "scene.json"
        [[nodiscard]] std::string displayName() const;
        // Same, with a trailing '*' while there are unsaved changes.
        [[nodiscard]] std::string title() const;

        // Fresh empty scene, no path, not dirty. The starter content is the
        // application's business, not the document's.
        void newScene();

        bool open(const std::filesystem::path& path, std::string* error = nullptr);

        // The same parse, from text the caller already has — a scene compiled
        // into the binary (see the editor's shipped examples), a document that
        // came over a socket. Everything after the parse is open()'s: the same
        // adopt-a-non-Scene-root rule, the same listeners, the same warnings.
        // The document has NO PATH afterwards, so it is untitled and the first
        // Save asks where to put it; that is the whole difference, and it is
        // what stops "open the example" from meaning "overwrite the example".
        bool openJson(const std::string& jsonText, std::string* error = nullptr);
        bool save(std::string* error = nullptr);
        bool saveAs(const std::filesystem::path& path, std::string* error = nullptr);

        // How much of the scene the file carries itself. The default embeds
        // everything, which is what a document you might hand to someone else
        // wants; referencing trades that for files that save and open in a
        // fraction of the time, at the cost of needing the source assets
        // alongside. Persisted in the editor's settings, not in the document —
        // it describes how you work, not what the scene is.
        [[nodiscard]] ImageStorage imageStorage() const { return imageStorage_; }
        void setImageStorage(ImageStorage storage) { imageStorage_ = storage; }

        [[nodiscard]] ModelStorage modelStorage() const { return modelStorage_; }
        void setModelStorage(ModelStorage storage) { modelStorage_ = storage; }

        // The document as it would be written, without touching the filesystem.
        [[nodiscard]] std::string toJson(bool prettyPrint, std::string* error = nullptr);

        // Swap in a different scene (a reload, or the restored play snapshot).
        // Registered editor-only objects are re-attached to the new scene and
        // listeners are fired. Does not change the path or the dirty flag.
        void replaceScene(std::shared_ptr<Scene> scene);

        // Fired after replaceScene(). The application uses this to re-point its
        // camera controls, re-resolve the selection by uuid and reattach helpers.
        int onSceneReplaced(std::function<void(Scene&)> listener);
        void removeListener(int id);

        // --- editor-only objects ------------------------------------------------
        // Attached to the scene by reference (the document never owns them) and
        // detached for the duration of any export.
        void addEditorOnly(Object3D& object);
        void removeEditorOnly(Object3D& object);
        [[nodiscard]] bool isEditorOnly(const Object3D& object) const;

        // --- play snapshots -----------------------------------------------------
        bool capture(SceneSnapshot& snapshot, std::string* error = nullptr);
        bool restore(const SceneSnapshot& snapshot, std::string* error = nullptr);

        // Warnings produced by the last export/import ("skipping userData entry
        // ...", "texture has no CPU-side image"). Shown in the console panel.
        [[nodiscard]] const std::vector<std::string>& warnings() const { return warnings_; }

    private:
        void detachEditorOnly();
        void attachEditorOnly();

        std::shared_ptr<Scene> scene_;
        std::filesystem::path path_;
        ImageStorage imageStorage_ = ImageStorage::Embed;
        ModelStorage modelStorage_ = ModelStorage::Embed;
        bool dirty_ = false;
        std::vector<Object3D*> editorOnly_;
        std::vector<std::string> warnings_;

        struct Listener {
            int id;
            std::function<void(Scene&)> fn;
        };
        std::vector<Listener> listeners_;
        int nextListenerId_ = 1;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SCENEDOCUMENT_HPP
