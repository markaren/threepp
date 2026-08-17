// The concrete commands the scene editor pushes onto a CommandStack.
//
// Every one of them stores BOTH sides of the change up front. None of them
// reads live scene state inside undo()/redo(), which is what makes them safe to
// replay in any order the stack chooses.
//
// A play-stop restore (or reload) replaces the WHOLE graph with an equivalent
// one, leaving stored Object3D pointers dangling. Uuids survive that round
// trip, so the commands here that target scene nodes re-resolve themselves in
// rebind(); the ones that cannot (type-erased setters capturing raw pointers)
// keep the base-class default and are dropped by CommandStack::rebind().

#ifndef THREEPP_EDITOR_EDITORCOMMANDS_HPP
#define THREEPP_EDITOR_EDITORCOMMANDS_HPP

#include "threepp/extras/editor/Command.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace threepp {

    class Material;
    class Texture;

}// namespace threepp

namespace threepp::editor {

    // Re-attach `child` to `parent` at a specific position among its children.
    //
    // Object3D::add() only appends, so restoring a deleted object's original
    // sibling order means detaching everything after the insertion point and
    // putting it back. Children attached with addRef() (helpers, gizmos) are
    // preserved as references, not adopted.
    void insertChildAt(Object3D& parent, const std::shared_ptr<Object3D>& child, std::size_t index);

    // Index of `child` in parent.children, or parent.children.size() if absent.
    [[nodiscard]] std::size_t childIndex(const Object3D& parent, const Object3D& child);

    // True when `candidate` is `object` or lives somewhere below it. Guards
    // reparent operations against building a cycle.
    [[nodiscard]] bool isDescendantOf(const Object3D& candidate, const Object3D& object);

    // Depth-first search for the node carrying `uuid`, or nullptr. Uuids are
    // stable across the play snapshot round trip, which makes them the one
    // usable identity after a scene replace.
    [[nodiscard]] Object3D* findByUuid(Object3D& root, const std::string& uuid);


    // ---------------------------------------------------------------- transform

    // Position / rotation / scale as one step. The gizmo and the three inspector
    // drag fields all produce this, so a rotate-then-move reads as two entries
    // rather than six.
    class SetTransformCommand: public Command {

    public:
        struct Trs {
            Vector3 position;
            Quaternion quaternion;
            Vector3 scale{1, 1, 1};
        };

        static Trs read(const Object3D& object);

        SetTransformCommand(Object3D& object, Trs before, Trs after, std::string label = "Transform");

        void redo() override;
        void undo() override;

        [[nodiscard]] std::string name() const override { return label_; }
        [[nodiscard]] std::string mergeKey() const override { return mergeKey_; }
        bool mergeWith(const Command& newer) override;
        [[nodiscard]] bool rebind(Object3D& root) override;

    private:
        void apply(const Trs& trs);

        Object3D* object_;
        std::string uuid_;
        Trs before_;
        Trs after_;
        std::string label_;
        std::string mergeKey_;
    };


    // ----------------------------------------------------------------- property

    // One field, one value, one setter. Covers everything in the inspector that
    // is not a transform: names, flags, colors, material scalars, light and
    // camera parameters.
    //
    // The setter is what makes this generic — it also carries any side effect the
    // field needs (material->needsUpdate(), camera->updateProjectionMatrix()).
    // It is also what ties the command to one scene: the lambda captures raw
    // pointers that cannot be re-resolved by uuid, so this command keeps the
    // default rebind() and is dropped when the scene is replaced.
    template<class T>
    class PropertyCommand: public Command {

    public:
        PropertyCommand(std::string label,
                        std::string mergeKey,
                        std::function<void(const T&)> setter,
                        T before,
                        T after)
            : label_(std::move(label)),
              mergeKey_(std::move(mergeKey)),
              setter_(std::move(setter)),
              before_(std::move(before)),
              after_(std::move(after)) {}

        void redo() override { setter_(after_); }
        void undo() override { setter_(before_); }

        [[nodiscard]] std::string name() const override { return label_; }
        [[nodiscard]] std::string mergeKey() const override { return mergeKey_; }

        bool mergeWith(const Command& newer) override {
            const auto* other = dynamic_cast<const PropertyCommand<T>*>(&newer);
            if (!other) return false;
            after_ = other->after_;
            return true;
        }

    private:
        std::string label_;
        std::string mergeKey_;
        std::function<void(const T&)> setter_;
        T before_;
        T after_;
    };

    template<class T>
    std::unique_ptr<Command> makeProperty(std::string label,
                                          std::string mergeKey,
                                          std::function<void(const T&)> setter,
                                          T before,
                                          T after) {

        return std::make_unique<PropertyCommand<T>>(
                std::move(label), std::move(mergeKey), std::move(setter),
                std::move(before), std::move(after));
    }


    // ------------------------------------------------------------- graph edits

    // Add a new object. The command owns the object while it is undone, so an
    // undone "Add Box" keeps the box (and its uuid) alive for redo.
    class AddObjectCommand: public Command {

    public:
        // Appends by default; `index` places the child at a specific position
        // among its siblings, which is what an "insert before/after" needs
        // (spline control points are ordered by exactly that).
        static constexpr std::size_t atEnd = static_cast<std::size_t>(-1);

        AddObjectCommand(Object3D& parent, std::shared_ptr<Object3D> object,
                         std::string label = "Add Object", std::size_t index = atEnd);

        void redo() override;
        void undo() override;

        [[nodiscard]] std::string name() const override { return label_; }
        bool mergeWith(const Command&) override { return false; }
        [[nodiscard]] bool rebind(Object3D& root) override;
        void retainedRoots(std::vector<Object3D*>& out) const override;

        [[nodiscard]] Object3D* object() const { return object_.get(); }

    private:
        Object3D* parent_;
        std::shared_ptr<Object3D> object_;
        std::string parentUuid_;
        std::string objectUuid_;
        std::string label_;
        std::size_t index_;
    };

    // Remove an object, retaining the subtree plus its parent and sibling index
    // so undo puts it back exactly where it was.
    class RemoveObjectCommand: public Command {

    public:
        explicit RemoveObjectCommand(Object3D& object, std::string label = "Delete Object");

        void redo() override;
        void undo() override;

        [[nodiscard]] std::string name() const override { return label_; }
        bool mergeWith(const Command&) override { return false; }
        [[nodiscard]] bool rebind(Object3D& root) override;
        void retainedRoots(std::vector<Object3D*>& out) const override;

        [[nodiscard]] Object3D* object() const { return raw_; }
        // Checked before execution: object_ (the retained ownership) is only
        // populated by redo(), so validity must rest on the raw target.
        [[nodiscard]] bool valid() const { return parent_ != nullptr && raw_ != nullptr; }

    private:
        Object3D* parent_;
        Object3D* raw_;
        std::string parentUuid_;
        std::string objectUuid_;
        std::shared_ptr<Object3D> object_;
        std::size_t index_;
        std::string label_;
    };

    // Move a subtree to a new parent, keeping its WORLD transform. The local
    // transform therefore changes; both sides are stored so undo is exact.
    class ReparentCommand: public Command {

    public:
        ReparentCommand(Object3D& object, Object3D& newParent, std::string label = "Reparent");

        void redo() override;
        void undo() override;

        [[nodiscard]] std::string name() const override { return label_; }
        bool mergeWith(const Command&) override { return false; }
        [[nodiscard]] bool rebind(Object3D& root) override;
        void retainedRoots(std::vector<Object3D*>& out) const override;

        // False when the move is impossible (no parent, or the target is a
        // descendant of the object). Callers must not push an invalid command.
        [[nodiscard]] bool valid() const { return valid_; }

    private:
        Object3D* object_;
        std::shared_ptr<Object3D> owned_;
        Object3D* oldParent_;
        Object3D* newParent_;
        std::string objectUuid_;
        std::string oldParentUuid_;
        std::string newParentUuid_;
        std::size_t oldIndex_;
        Vector3 oldPosition_, newPosition_;
        Quaternion oldQuaternion_, newQuaternion_;
        Vector3 oldScale_, newScale_;
        std::string label_;
        bool valid_ = false;
    };


    // ------------------------------------------------------------------ texture

    // Assign or clear one texture slot of a material. Kept separate from
    // PropertyCommand so the "which slot" name survives into the undo label and
    // the material's needsUpdate() is never forgotten. Like PropertyCommand it
    // holds raw pointers a scene replace invalidates, keeps the default
    // rebind(), and is dropped from the stack when that happens.
    class SetMaterialMapCommand: public Command {

    public:
        SetMaterialMapCommand(Material& material,
                              std::string slot,
                              std::function<void(const std::shared_ptr<Texture>&)> setter,
                              std::shared_ptr<Texture> before,
                              std::shared_ptr<Texture> after);

        void redo() override;
        void undo() override;

        [[nodiscard]] std::string name() const override { return label_; }
        bool mergeWith(const Command&) override { return false; }

    private:
        void apply(const std::shared_ptr<Texture>& texture);

        Material* material_;
        std::function<void(const std::shared_ptr<Texture>&)> setter_;
        std::shared_ptr<Texture> before_;
        std::shared_ptr<Texture> after_;
        std::string label_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_EDITORCOMMANDS_HPP
