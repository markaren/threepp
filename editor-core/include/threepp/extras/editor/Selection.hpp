// What the editor is currently pointed at.
//
// Single-select: the hierarchy, the inspector, the gizmo and the shortcut
// handlers all read this one object, and any of them may set it. Listeners let
// them stay in sync without knowing about each other.

#ifndef THREEPP_EDITOR_SELECTION_HPP
#define THREEPP_EDITOR_SELECTION_HPP

#include <functional>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    class Selection {

    public:
        // Non-owning: the scene graph owns the object. Whoever removes an object
        // from the scene is responsible for clearing (or re-pointing) the
        // selection first — see SceneDocument::replaceScene, which re-resolves
        // by uuid.
        [[nodiscard]] Object3D* get() const { return current_; }

        [[nodiscard]] bool empty() const { return current_ == nullptr; }

        // uuid of the selected object, "" when nothing is selected. Survives a
        // scene reload, which raw pointers do not.
        [[nodiscard]] std::string uuid() const;

        void set(Object3D* object);

        void clear() { set(nullptr); }

        // Listeners fire only on an actual change, and receive the new selection
        // (nullptr when cleared).
        int onChange(std::function<void(Object3D*)> listener);

        void removeListener(int id);

    private:
        struct Entry {
            int id;
            std::function<void(Object3D*)> fn;
        };

        Object3D* current_ = nullptr;
        // Captured on set() rather than read back from current_. Re-resolving a
        // selection is needed exactly when the old graph is gone, and
        // SceneDocument::replaceScene frees it before it notifies anyone.
        std::string uuid_;
        std::vector<Entry> listeners_;
        int nextId_ = 1;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SELECTION_HPP
