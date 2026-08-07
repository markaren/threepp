
#include "threepp/extras/editor/EditorCommands.hpp"

#include "threepp/materials/Material.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <iterator>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A child detached while making room for an insertion. Children added with
    // add() come back as an owning pointer; children added with addRef() have no
    // owning pointer here and are re-attached by reference.
    struct Detached {
        std::shared_ptr<Object3D> owned;
        Object3D* raw;
    };

}// namespace


void threepp::editor::insertChildAt(Object3D& parent, const std::shared_ptr<Object3D>& child, std::size_t index) {

    if (!child) return;

    if (index >= parent.children.size()) {
        parent.add(child);
        return;
    }

    std::vector<Detached> tail;
    tail.reserve(parent.children.size() - index);
    while (parent.children.size() > index) {
        Object3D* last = parent.children.back();
        auto owned = last->removeFromParent();
        tail.push_back({owned, last});
    }

    parent.add(child);

    for (auto it = tail.rbegin(); it != tail.rend(); ++it) {
        if (it->owned) {
            parent.add(it->owned);
        } else {
            parent.addRef(*it->raw);
        }
    }
}

std::size_t threepp::editor::childIndex(const Object3D& parent, const Object3D& child) {

    const auto it = std::find(parent.children.begin(), parent.children.end(), &child);
    return static_cast<std::size_t>(std::distance(parent.children.begin(), it));
}

bool threepp::editor::isDescendantOf(const Object3D& candidate, const Object3D& object) {

    for (const Object3D* o = &candidate; o != nullptr; o = o->parent) {
        if (o == &object) return true;
    }
    return false;
}

Object3D* threepp::editor::findByUuid(Object3D& root, const std::string& uuid) {

    if (root.uuid == uuid) return &root;
    for (auto* child : root.children) {
        if (auto* found = findByUuid(*child, uuid)) return found;
    }
    return nullptr;
}


// ------------------------------------------------------------------ transform

SetTransformCommand::Trs SetTransformCommand::read(const Object3D& object) {

    Trs trs;
    trs.position.copy(object.position);
    trs.quaternion.copy(object.quaternion);
    trs.scale.copy(object.scale);
    return trs;
}

SetTransformCommand::SetTransformCommand(Object3D& object, Trs before, Trs after, std::string label)
    : object_(&object),
      uuid_(object.uuid),
      before_(before),
      after_(after),
      label_(std::move(label)),
      mergeKey_("transform:" + object.uuid) {}

void SetTransformCommand::apply(const Trs& trs) {

    object_->position.copy(trs.position);
    object_->quaternion.copy(trs.quaternion);
    object_->scale.copy(trs.scale);
    // Quaternion::copy fires Object3D's change hook, which keeps the Euler
    // `rotation` (what the inspector shows) in step — no manual sync needed.
    object_->updateMatrix();
    object_->matrixWorldNeedsUpdate = true;
}

void SetTransformCommand::redo() {

    apply(after_);
}

void SetTransformCommand::undo() {

    apply(before_);
}

bool SetTransformCommand::mergeWith(const Command& newer) {

    const auto* other = dynamic_cast<const SetTransformCommand*>(&newer);
    if (!other || other->object_ != object_) return false;
    after_ = other->after_;
    label_ = other->label_;
    return true;
}

bool SetTransformCommand::rebind(Object3D& root) {

    auto* found = findByUuid(root, uuid_);
    if (!found) return false;
    object_ = found;
    return true;
}


// ----------------------------------------------------------------- graph edits

AddObjectCommand::AddObjectCommand(Object3D& parent, std::shared_ptr<Object3D> object,
                                   std::string label, std::size_t index)
    : parent_(&parent),
      object_(std::move(object)),
      parentUuid_(parent.uuid),
      objectUuid_(object_ ? object_->uuid : std::string{}),
      label_(std::move(label)),
      index_(std::min(index, parent.children.size())) {}

void AddObjectCommand::redo() {

    if (!object_) return;
    insertChildAt(*parent_, object_, index_);
}

void AddObjectCommand::undo() {

    if (!object_) return;
    index_ = childIndex(*parent_, *object_);
    object_->removeFromParent();
}

void AddObjectCommand::retainedRoots(std::vector<Object3D*>& out) const {

    // Held whether the add is applied or undone; the stack charges it to the
    // history only in the undone state, where the object is out of the scene and
    // this shared_ptr is the last thing keeping it alive.
    if (object_) out.push_back(object_.get());
}

bool AddObjectCommand::rebind(Object3D& root) {

    auto* parent = findByUuid(root, parentUuid_);
    if (!parent || !object_) return false;
    parent_ = parent;

    // The added object usually came back as a fresh instance carrying the same
    // uuid; undo must remove THAT one. When it is absent from the new graph
    // (the add is currently undone, or a later delete removed it again), the
    // instance retained here stays the target — the shared_ptr keeps it alive
    // and ~Object3D nulled its parent when the old scene died.
    if (auto* live = findByUuid(root, objectUuid_)) {
        auto owned = live->weak_from_this().lock();
        if (!owned) return false;
        object_ = std::move(owned);
    }
    return true;
}


RemoveObjectCommand::RemoveObjectCommand(Object3D& object, std::string label)
    : parent_(object.parent),
      raw_(&object),
      parentUuid_(parent_ ? parent_->uuid : std::string{}),
      objectUuid_(object.uuid),
      index_(0),
      label_(std::move(label)) {

    if (parent_) index_ = childIndex(*parent_, object);
}

void RemoveObjectCommand::redo() {

    if (!parent_ || !raw_) return;
    index_ = childIndex(*parent_, *raw_);
    // Holds the only remaining reference while the object is out of the tree.
    // Null for a child that was attached by reference — then the owner is
    // elsewhere and undo re-attaches by reference too.
    object_ = raw_->removeFromParent();
}

void RemoveObjectCommand::undo() {

    if (!parent_ || !raw_) return;
    if (object_) {
        insertChildAt(*parent_, object_, index_);
        object_.reset();
    } else {
        parent_->addRef(*raw_);
    }
}

void RemoveObjectCommand::retainedRoots(std::vector<Object3D*>& out) const {

    // Non-null exactly while the deletion stands (redo() takes ownership, undo()
    // hands it back), which is precisely when the deleted subtree's memory is
    // the history's to account for.
    if (object_) out.push_back(object_.get());
}

bool RemoveObjectCommand::rebind(Object3D& root) {

    auto* parent = findByUuid(root, parentUuid_);
    if (!parent) return false;
    parent_ = parent;

    // Applied: the removed subtree is out of every scene and owned right here,
    // so raw_ still points at a live node and undo reinserts it as-is.
    if (object_) return true;

    // Undone: the object is back in the (now replaced) graph.
    auto* live = findByUuid(root, objectUuid_);
    if (!live) return false;
    raw_ = live;
    return true;
}


ReparentCommand::ReparentCommand(Object3D& object, Object3D& newParent, std::string label)
    : object_(&object),
      oldParent_(object.parent),
      newParent_(&newParent),
      oldIndex_(0),
      label_(std::move(label)) {

    if (!oldParent_) return;
    if (oldParent_ == newParent_) return;
    // Moving a node under one of its own descendants would detach the subtree
    // from the scene entirely.
    if (isDescendantOf(newParent, object)) return;

    oldIndex_ = childIndex(*oldParent_, object);

    oldPosition_.copy(object.position);
    oldQuaternion_.copy(object.quaternion);
    oldScale_.copy(object.scale);

    // World transform is preserved across the move, so the object does not jump
    // when it is dropped onto a transformed parent. That means a NEW local
    // transform, computed here once and stored like any other command state.
    object.updateWorldMatrix(true, false);
    newParent.updateWorldMatrix(true, false);

    Matrix4 inverseParent;
    inverseParent.copy(*newParent.matrixWorld).invert();
    Matrix4 local;
    local.multiplyMatrices(inverseParent, *object.matrixWorld);
    local.decompose(newPosition_, newQuaternion_, newScale_);

    objectUuid_ = object.uuid;
    oldParentUuid_ = oldParent_->uuid;
    newParentUuid_ = newParent.uuid;

    valid_ = true;
}

void ReparentCommand::redo() {

    if (!valid_) return;

    owned_ = object_->removeFromParent();
    if (owned_) {
        newParent_->add(owned_);
    } else {
        newParent_->addRef(*object_);
    }
    object_->position.copy(newPosition_);
    object_->quaternion.copy(newQuaternion_);
    object_->scale.copy(newScale_);
    object_->updateMatrix();
    object_->matrixWorldNeedsUpdate = true;
}

void ReparentCommand::undo() {

    if (!valid_) return;

    auto owned = object_->removeFromParent();
    if (owned) {
        insertChildAt(*oldParent_, owned, oldIndex_);
    } else {
        oldParent_->addRef(*object_);
    }
    owned_.reset();
    object_->position.copy(oldPosition_);
    object_->quaternion.copy(oldQuaternion_);
    object_->scale.copy(oldScale_);
    object_->updateMatrix();
    object_->matrixWorldNeedsUpdate = true;
}

void ReparentCommand::retainedRoots(std::vector<Object3D*>& out) const {

    // Only ever held while the subtree sits under its new parent, so this
    // normally costs the history nothing — reported anyway rather than assumed,
    // since a reparent under a subtree that is ITSELF detached is exactly the
    // case an assumption would get wrong.
    if (owned_) out.push_back(owned_.get());
}

bool ReparentCommand::rebind(Object3D& root) {

    if (!valid_) return false;

    // All three must be live in the new graph; the moved object cannot ride on
    // a retained subtree here because this command never owns one itself.
    auto* object = findByUuid(root, objectUuid_);
    auto* oldParent = findByUuid(root, oldParentUuid_);
    auto* newParent = findByUuid(root, newParentUuid_);
    if (!object || !oldParent || !newParent) return false;

    object_ = object;
    oldParent_ = oldParent;
    newParent_ = newParent;
    // Any ownership held from before the swap refers to the dead graph's
    // instance; redo/undo re-obtain it from the new parent via
    // removeFromParent().
    owned_.reset();
    return true;
}


// -------------------------------------------------------------------- texture

SetMaterialMapCommand::SetMaterialMapCommand(Material& material,
                                             std::string slot,
                                             std::function<void(const std::shared_ptr<Texture>&)> setter,
                                             std::shared_ptr<Texture> before,
                                             std::shared_ptr<Texture> after)
    : material_(&material),
      setter_(std::move(setter)),
      before_(std::move(before)),
      after_(std::move(after)),
      label_((after_ ? "Set " : "Clear ") + slot) {}

void SetMaterialMapCommand::apply(const std::shared_ptr<Texture>& texture) {

    setter_(texture);
    // Adding or removing a map changes the shader permutation, not just a
    // uniform — without this the renderer keeps the old program.
    material_->needsUpdate();
}

void SetMaterialMapCommand::redo() {

    apply(after_);
}

void SetMaterialMapCommand::undo() {

    apply(before_);
}
