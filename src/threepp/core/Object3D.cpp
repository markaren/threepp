
#include "threepp/core/Object3D.hpp"

#include "threepp/animation/AnimationClip.hpp"

#include "threepp/cameras/Camera.hpp"

#include "threepp/math/MathUtils.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Raycaster.hpp"

#include "threepp/lights/Light.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

using namespace threepp;

namespace {

    // Gate for add()/addRef(): inserting a node under itself or under one of
    // its own descendants makes the graph cyclic, and traverse()/
    // updateMatrixWorld() recurse over children without cycle detection - so
    // the insertion is rejected here instead of overflowing the stack there.
    bool canAttach(const Object3D& parent, const Object3D& child) {

        for (const auto* node = &parent; node; node = node->parent) {

            if (node == &child) {

                std::cerr << "[Object3D] add: rejected inserting '" << child.name
                          << "' under '" << parent.name
                          << "' - it is the target itself or one of its ancestors" << std::endl;
                return false;
            }
        }

        return true;
    }

}// namespace

Object3D::Object3D()
    : uuid(math::generateUUID()) {

    rotation._onChange([this] {
        quaternion.setFromEuler(rotation, false);
    });
    quaternion._onChange([this] {
        rotation.setFromQuaternion(quaternion, std::nullopt, false);
    });
}

std::string Object3D::type() const {

    return "Object3D";
}

void Object3D::applyMatrix4(const Matrix4& m) {

    if (this->matrixAutoUpdate) this->updateMatrix();

    this->matrix->premultiply(m);

    this->matrix->decompose(this->position, this->quaternion, this->scale);
}

Object3D& Object3D::applyQuaternion(const Quaternion& q) {

    this->quaternion.premultiply(q);

    return *this;
}

void Object3D::setRotationFromAxisAngle(const Vector3& axis, float angle) {

    // assumes axis is normalized

    this->quaternion.setFromAxisAngle(axis, angle);
}

void Object3D::setRotationFromEuler(const Euler& euler) {

    this->quaternion.setFromEuler(euler, true);
}

void Object3D::setRotationFromMatrix(const Matrix4& m) {

    // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

    this->quaternion.setFromRotationMatrix(m);
}

void Object3D::setRotationFromQuaternion(const Quaternion& q) {

    // assumes q is normalized

    this->quaternion = q;
}

Object3D& Object3D::rotateOnAxis(const Vector3& axis, float angle) {

    // rotate object on axis in object space
    // axis is assumed to be normalized

    Quaternion _q1{};

    _q1.setFromAxisAngle(axis, angle);

    this->quaternion.multiply(_q1);

    return *this;
}

Object3D& Object3D::rotateOnWorldAxis(const Vector3& axis, float angle) {

    // rotate object on axis in world space
    // axis is assumed to be normalized
    // method assumes no rotated parent

    Quaternion _q1{};

    _q1.setFromAxisAngle(axis, angle);

    this->quaternion.premultiply(_q1);

    return *this;
}

Object3D& Object3D::rotateX(float angle) {

    return this->rotateOnAxis(Vector3::X(), angle);
}

Object3D& Object3D::rotateY(float angle) {

    return this->rotateOnAxis(Vector3::Y(), angle);
}

Object3D& Object3D::rotateZ(float angle) {

    return this->rotateOnAxis(Vector3::Z(), angle);
}

Object3D& Object3D::translateOnAxis(const Vector3& axis, float distance) {

    // translate object by distance along axis in object space
    // axis is assumed to be normalized

    Vector3 _v1{};

    _v1.copy(axis).applyQuaternion(this->quaternion);

    this->position.add(_v1.multiplyScalar(distance));

    return *this;
}

Object3D& Object3D::translateX(float distance) {

    return this->translateOnAxis(Vector3::X(), distance);
}

Object3D& Object3D::translateY(float distance) {

    return this->translateOnAxis(Vector3::Y(), distance);
}

Object3D& Object3D::translateZ(float distance) {

    return this->translateOnAxis(Vector3::Z(), distance);
}

void Object3D::localToWorld(Vector3& vector) {

    this->updateWorldMatrix(true, false);// https://github.com/mrdoob/three.js/pull/25097

    vector.applyMatrix4(*this->matrixWorld);
}

void Object3D::worldToLocal(Vector3& vector) {

    this->updateWorldMatrix(true, false);// https://github.com/mrdoob/three.js/pull/25097

    Matrix4 _m1{};

    vector.applyMatrix4(_m1.copy(*this->matrixWorld).invert());
}

void Object3D::lookAt(const Vector3& vector) {

    lookAt(vector.x, vector.y, vector.z);
}

void Object3D::lookAt(float x, float y, float z) {

    // This method does not support objects having non-uniformly-scaled parent(s)

    Vector3 _target{};
    Vector3 _position{};
    Quaternion _q1{};
    Matrix4 _m1{};

    _target.set(x, y, z);

    this->updateWorldMatrix(true, false);

    _position.setFromMatrixPosition(*this->matrixWorld);

    if (this->is<Camera>() || this->is<Light>() || this->usesCameraLookAtConvention()) {

        _m1.lookAt(_position, _target, this->up);

    } else {

        _m1.lookAt(_target, _position, this->up);
    }

    this->quaternion.setFromRotationMatrix(_m1);

    if (parent) {

        _m1.extractRotation(*parent->matrixWorld);
        _q1.setFromRotationMatrix(_m1);
        this->quaternion.premultiply(_q1.invert());
    }
}

void Object3D::add(const std::shared_ptr<Object3D>& object) {

    if (!object) {

        std::cerr << "[Object3D] add: ignored null child" << std::endl;
        return;
    }

    addRef(*object);

    // addRef() validates the insertion; only take ownership when the link was
    // actually made - a rejected self/ancestor insertion must not be kept
    // alive by a parent that does not list it as a child.
    if (object->parent != this) return;

    // Reparenting a child its old parent owned: addRef() already moved that
    // owning reference into children_ - don't own it twice.
    if (std::ranges::none_of(children_, [&object](const auto& c) { return c == object; })) {

        this->children_.emplace_back(object);
    }
}

void Object3D::addRef(Object3D& object) {

    if (!canAttach(*this, object)) return;

    // Unlink from any current parent directly rather than via remove():
    // remove() drops the old parent's owning reference, destroying an object
    // that parent solely owned - while this function still has to write to it.
    std::shared_ptr<Object3D> kept;
    if (object.parent) {

        kept = object.parent->detachChild(object);
    }

    object.parent = this;
    this->children.emplace_back(&object);

    // The new parent's world transform must flow into this subtree even when
    // the child's local transform is unchanged (updateMatrix()'s early-out no
    // longer raises this flag every frame). The child's world multiply then
    // force-propagates to its descendants.
    object.matrixWorldNeedsUpdate = true;

    // Ownership follows the object: if the old parent owned it, letting `kept`
    // die here would destroy the object the moment this call returned.
    if (kept) {

        this->children_.emplace_back(std::move(kept));
    }

    object.dispatchEvent("added");
}

std::shared_ptr<Object3D> Object3D::detachChild(Object3D& object) {

    // Take the owning reference out of children_ FIRST, into a local. Erasing it
    // in place would run the shared_ptr destructor right here — destroying the
    // object while we still have to clear its parent and fire "remove", and while
    // the caller still holds the reference it passed in. Moving it out keeps the
    // object alive until the caller drops what we return.
    std::shared_ptr<Object3D> owned;
    if (const auto it = std::ranges::find_if(children_, [&object](const auto& obj) {
            return obj.get() == &object;
        });
        it != children_.end()) {

        owned = std::move(*it);
        children_.erase(it);
    }

    // non-owning (all children should be represented here)
    if (const auto it = std::ranges::find_if(children, [&object](const auto& obj) {
            return obj == &object;
        });
        it != children.end()) {

        Object3D* child = *it;
        children.erase(it);

        child->parent = nullptr;
        child->dispatchEvent("remove", child);
    }

    return owned;
}

void Object3D::remove(Object3D& object) {

    // The returned reference dies at the end of this statement, so an object the
    // parent solely owned is destroyed here — but only after it has been fully
    // unlinked and its listeners have run against a live object.
    detachChild(object);
}

std::shared_ptr<Object3D> Object3D::removeFromParent() {

    if (!parent) return nullptr;

    // Hand the owning reference to the caller rather than dropping it inside
    // this call: `parent->remove(*this)` used to free `this` while this frame was
    // still executing, which is UB even though nothing touched a member
    // afterwards. Now self-removal is safe, and `auto kept = o->removeFromParent()`
    // is the way to detach without destroying.
    return parent->detachChild(*this);
}

void Object3D::clear() {

    for (auto& object : this->children) {

        object->parent = nullptr;

        object->dispatchEvent("remove");
    }

    this->children.clear();
    this->children_.clear();
}

void Object3D::getWorldPosition(Vector3& target) {

    this->updateWorldMatrix(true, false);

    target.setFromMatrixPosition(*this->matrixWorld);
}

void Object3D::getWorldQuaternion(Quaternion& target) {

    Vector3 _position{};
    Vector3 _scale{};

    this->updateWorldMatrix(true, false);

    this->matrixWorld->decompose(_position, target, _scale);
}

void Object3D::getWorldScale(Vector3& target) {

    Vector3 _position{};
    Quaternion _quaternion{};

    this->updateWorldMatrix(true, false);

    this->matrixWorld->decompose(_position, _quaternion, target);
}

void Object3D::getWorldDirection(Vector3& target) {

    this->updateWorldMatrix(true, false);

    const auto& e = this->matrixWorld->elements;

    target.set(e[8], e[9], e[10]).normalize();
}

void Object3D::traverse(const std::function<void(Object3D&)>& callback) {

    callback(*this);

    for (auto& c : children) {

        c->traverse(callback);
    }
}

void Object3D::traverseVisible(const std::function<void(Object3D&)>& callback) {

    if (!this->visible) return;

    callback(*this);

    for (auto& c : children) {

        c->traverseVisible(callback);
    }
}

void Object3D::traverseAncestors(const std::function<void(Object3D&)>& callback) {

    if (parent) {

        callback(*parent);

        parent->traverseAncestors(callback);
    }
}

void Object3D::updateMatrix() {

    // Early-out when nothing moved since the last compose. matrixAutoUpdate
    // recomposes EVERY object EVERY frame, so a large static scene paid
    // thousands of composes plus the cascading world multiplies (the compose
    // unconditionally raised matrixWorldNeedsUpdate) — several ms/frame of
    // pure CPU on scenes like Bistro. Comparing the source values keeps the
    // three.js polling contract bit-identical: mutations to position/
    // rotation/quaternion/scale are picked up on the very next frame with no
    // user-side notification; comparing the composed matrix bytes too means a
    // direct user write to `matrix` while matrixAutoUpdate is on still gets
    // clobbered by the recompose, exactly as before. (NaN compares unequal,
    // so degenerate values safely fall through to the recompose.)
    const std::array<float, 10> pqs{position.x, position.y, position.z,
                                    quaternion.x, quaternion.y, quaternion.z, quaternion.w,
                                    scale.x, scale.y, scale.z};
    if (composedValid_ && pqs == composedPqs_ &&
        std::memcmp(matrix->elements.data(), composedMatrix_.data(), sizeof(composedMatrix_)) == 0) {
        return;// unchanged — matrixWorldNeedsUpdate stays as-is, subtree multiply skipped
    }

    this->matrix->compose(this->position, this->quaternion, this->scale);

    composedPqs_ = pqs;
    std::memcpy(composedMatrix_.data(), matrix->elements.data(), sizeof(composedMatrix_));
    composedValid_ = true;

    this->matrixWorldNeedsUpdate = true;
}

void Object3D::updateMatrixWorld(bool force) {

    if (this->matrixAutoUpdate) {

        this->updateMatrix();

    } else if (!composedValid_ ||
               std::memcmp(matrix->elements.data(), composedMatrix_.data(), sizeof(composedMatrix_)) != 0) {

        // matrixAutoUpdate == false means `matrix` is driven externally:
        // helpers alias another object's matrixWorld (CameraHelper, the light
        // helpers), loaders bake static transforms, users write it directly.
        // These writes used to propagate only because the unconditional
        // every-frame compose at the scene root force-cascaded the world
        // multiply to every descendant; with updateMatrix()'s early-out that
        // cascade is gone, so poll the matrix bytes here instead. Mutations
        // still show up on the very next frame, unchanged matrices still skip
        // the multiply.
        std::memcpy(composedMatrix_.data(), matrix->elements.data(), sizeof(composedMatrix_));
        // Poison the PQS snapshot: if matrixAutoUpdate is re-enabled, the
        // external matrix must be clobbered by a recompose (NaN never
        // compares equal, so updateMatrix() cannot early-out on it).
        composedPqs_.fill(std::numeric_limits<float>::quiet_NaN());
        composedValid_ = true;

        this->matrixWorldNeedsUpdate = true;
    }

    if (this->matrixWorldNeedsUpdate || force) {

        if (!this->parent) {

            this->matrixWorld->copy(*this->matrix);

        } else {

            this->matrixWorld->multiplyMatrices(*this->parent->matrixWorld, *this->matrix);
        }

        this->matrixWorldNeedsUpdate = false;

        force = true;
    }

    // update children

    for (auto& child : this->children) {

        child->updateMatrixWorld(force);
    }
}

void Object3D::updateWorldMatrix(bool updateParents, bool updateChildren) {

    if (updateParents && parent) {

        parent->updateWorldMatrix(true, false);
    }

    if (this->matrixAutoUpdate) this->updateMatrix();

    if (!this->parent) {

        this->matrixWorld->copy(*this->matrix);

    } else {

        this->matrixWorld->multiplyMatrices(*this->parent->matrixWorld, *this->matrix);
    }

    // update children

    if (updateChildren) {

        for (const auto& child : children) {

            child->updateWorldMatrix(false, true);
        }
    }
}

void Object3D::copy(const Object3D& source, bool recursive) {

    this->name = source.name;

    this->up.copy(source.up);

    this->position.copy(source.position);
    this->rotation.order_ = source.rotation.order_;
    this->quaternion.copy(source.quaternion);
    this->scale.copy(source.scale);

    this->matrix->copy(*source.matrix);
    this->matrixWorld->copy(*source.matrixWorld);

    this->matrixAutoUpdate = source.matrixAutoUpdate;
    this->matrixWorldNeedsUpdate = source.matrixWorldNeedsUpdate;

    this->layers.mask_ = source.layers.mask_;
    this->visible = source.visible;

    this->castShadow = source.castShadow;
    this->receiveShadow = source.receiveShadow;

    this->frustumCulled = source.frustumCulled;
    this->renderOrder = source.renderOrder;

    // three.js copies userData too (Object3D.copy deep-clones it). Without this
    // a clone silently loses everything attached to the object by application
    // code - physics setup, gameplay tags, editor annotations - which is
    // exactly the data a duplicate is expected to carry.
    this->userData = source.userData;

    if (recursive) {

        for (const auto& child : source.children) {

            auto clone = child->clone();
            this->add(clone);
        }
    }
}

Object3D::Object3D(Object3D&& source) noexcept: Object3D() {

    this->name = std::move(source.name);

    this->up = source.up;
    source.up = defaultUp;

    // Take over the source's slot in its parent. Transferring only the parent
    // POINTER left the parent's child list still naming the moved-from object -
    // a dangling entry once the source died, and freed-memory reads on the
    // next traversal.
    if (Object3D* p = source.parent) {

        const bool ownedByParent = std::ranges::any_of(p->children_, [&source](const auto& c) {
            return c.get() == &source;
        });
        if (ownedByParent) {
            // The parent owns the source's storage through a shared_ptr, which
            // cannot follow a move to a new address. Leave the hollowed-out
            // source attached where its owner expects it; this object starts
            // detached.
        } else {
            this->parent = p;
            source.parent = nullptr;
            std::ranges::replace(p->children, &source, this);
        }
    }

    this->scale.copy(source.scale);
    this->position.copy(source.position);

    this->rotation = std::move(source.rotation);
    this->quaternion = std::move(source.quaternion);

    // The matrices live by value inside the object, so "moving" them copies
    // the 64-byte payloads into our own storage — which also leaves the
    // moved-from source with VALID matrices (it may still sit in its parent's
    // children list, see above; stealing the old shared_ptrs left it with
    // nulls and a guaranteed deref on the next traversal). When the source's
    // handle was re-pointed at external storage (helpers alias another
    // object's matrixWorld), transfer the alias itself instead.
    if (source.matrix.get() == &source.matrixLocal_) {
        this->matrixLocal_.copy(source.matrixLocal_);
    } else {
        this->matrix = source.matrix;
    }
    if (source.matrixWorld.get() == &source.matrixWorldLocal_) {
        this->matrixWorldLocal_.copy(source.matrixWorldLocal_);
    } else {
        this->matrixWorld = source.matrixWorld;
    }

    this->matrixAutoUpdate = source.matrixAutoUpdate;
    this->matrixWorldNeedsUpdate = source.matrixWorldNeedsUpdate;

    this->layers.mask_ = source.layers.mask_;
    this->visible = source.visible;

    this->castShadow = source.castShadow;
    this->receiveShadow = source.receiveShadow;

    this->frustumCulled = source.frustumCulled;
    this->renderOrder = source.renderOrder;

    this->onAfterRender = std::move(source.onAfterRender);
    this->onBeforeRender = std::move(source.onBeforeRender);

    this->rotation._onChange([this] {
        quaternion.setFromEuler(rotation, false);
    });
    this->quaternion._onChange([this] {
        rotation.setFromQuaternion(quaternion, std::nullopt, false);
    });

    this->children = std::move(source.children);
    this->children_ = std::move(source.children_);

    for (auto& c : children) {
        c->parent = this;
    }
}

Object3D::~Object3D() {

    // Leave no dangling references to this node.
    //
    // `children` is a vector of raw, non-owning pointers, and addRef() puts
    // objects there that the parent does NOT own. Such a child can easily die
    // first — a stack local, or a unique_ptr the caller holds — and until now the
    // parent kept a dangling entry, so the next traverse() was a use-after-free:
    //
    //     { Mesh m{geo, mat}; scene->addRef(m); }   // m dies here
    //     scene->traverse(...);                     // read of freed memory
    //
    // Detaching in the destructor makes that safe without changing addRef()'s
    // non-owning contract.
    if (parent) {
        auto& siblings = parent->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this), siblings.end());
        parent = nullptr;
    }

    // Symmetrically: any child we do not own outlives us, so clear its parent
    // before it can point at freed memory. Doing this in the destructor BODY also
    // means the owned children_ (destroyed after the body, in reverse member
    // order) already see parent == nullptr and skip the erase above — so they
    // never reach back into a half-destroyed parent.
    for (auto* child : children) {
        if (child->parent == this) child->parent = nullptr;
    }
    children.clear();
}
