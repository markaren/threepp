
#include "threepp/core/Object3D.hpp"

#include "threepp/animation/AnimationClip.hpp"

#include "threepp/cameras/Camera.hpp"

#include "threepp/math/MathUtils.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Raycaster.hpp"

#include "threepp/lights/Light.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

using namespace threepp;

Object3D::Object3D()
    : uuid(math::generateUUID()),
      matrix(std::make_shared<Matrix4>()),
      matrixWorld(std::make_shared<Matrix4>()) {

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

    if (this->is<Camera>() || this->is<Light>()) {

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

    this->children_.emplace_back(object);
    addRef(*object);
}

void Object3D::addRef(Object3D& object) {

    if (object.parent) {

        object.parent->remove(object);
    }

    object.parent = this;
    this->children.emplace_back(&object);

    // The new parent's world transform must flow into this subtree even when
    // the child's local transform is unchanged (updateMatrix()'s early-out no
    // longer raises this flag every frame). The child's world multiply then
    // force-propagates to its descendants.
    object.matrixWorldNeedsUpdate = true;

    object.dispatchEvent("added");
}

void Object3D::remove(Object3D& object) {

    // non-owning (all children should be represented here)
    if (const auto find = std::ranges::find_if(children, [&object](const auto& obj) {
            return obj == &object;
        });
        find != children.end()) {

        Object3D* child = *find;
        children.erase(find);

        child->parent = nullptr;
        child->dispatchEvent("remove", child);
    }

    // owning
    if (const auto find = std::ranges::find_if(children_, [&object](const auto& obj) {
            return obj.get() == &object;
        });
        find != children_.end()) {

        children_.erase(find);
    }
}

void Object3D::removeFromParent() {

    if (parent) {

        parent->remove(*this);
    }
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

    this->parent = source.parent;
    source.parent = nullptr;

    this->scale.copy(source.scale);
    this->position.copy(source.position);

    this->rotation = std::move(source.rotation);
    this->quaternion = std::move(source.quaternion);

    this->matrix = std::move(source.matrix);
    this->matrixWorld = std::move(source.matrixWorld);

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

Object3D::~Object3D() = default;
