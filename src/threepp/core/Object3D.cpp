
#include "threepp/core/Object3D.hpp"
#include "threepp/core/BufferGeometry.hpp"

#include "threepp/utils/InstanceOf.hpp"
#include "threepp/cameras/Camera.hpp"
#include "threepp/lights/Light.hpp"

using namespace threepp;

Vector3 Object3D::defaultUp = Vector3(0, 1, 0);

namespace {

    Vector3 _v1;
    Quaternion _q1;
    Matrix4 _m1;
    Vector3 _target;

    Vector3 _scale;
    Vector3 _position;
    Quaternion _quaternion;

}// namespace

unsigned int Object3D::_object3Did = 0;

void Object3D::applyMatrix4(const Matrix4 &matrix) {

    if (this->matrixAutoUpdate) this->updateMatrix();

    this->matrix.premultiply(matrix);

    this->matrix.decompose(this->position, this->quaternion, this->scale);
}

Object3D &Object3D::applyQuaternion(const Quaternion &q) {

    this->quaternion.premultiply(q);

    return *this;
}

void Object3D::setRotationFromAxisAngle(const Vector3 &axis, float angle) {

    // assumes axis is normalized

    this->quaternion.setFromAxisAngle(axis, angle);
}

void Object3D::setRotationFromEuler(const Euler &euler) {

    this->quaternion.setFromEuler(euler, true);
}

void Object3D::setRotationFromMatrix(const Matrix4 &m) {

    // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

    this->quaternion.setFromRotationMatrix(m);
}

void Object3D::setRotationFromQuaternion(const Quaternion &q) {

    // assumes q is normalized

    this->quaternion = q;
}

Object3D &Object3D::rotateOnAxis(const Vector3 &axis, float angle) {

    // rotate object on axis in object space
    // axis is assumed to be normalized

    _q1.setFromAxisAngle(axis, angle);

    this->quaternion.multiply(_q1);

    return *this;
}

Object3D &Object3D::rotateOnWorldAxis(const Vector3 &axis, float angle) {

    // rotate object on axis in world space
    // axis is assumed to be normalized
    // method assumes no rotated parent

    _q1.setFromAxisAngle(axis, angle);

    this->quaternion.premultiply(_q1);

    return *this;
}

Object3D &Object3D::rotateX(float angle) {

    return this->rotateOnAxis(Vector3::X, angle);
}

Object3D &Object3D::rotateY(float angle) {

    return this->rotateOnAxis(Vector3::Y, angle);
}

Object3D &Object3D::rotateZ(float angle) {

    return this->rotateOnAxis(Vector3::Z, angle);
}

Object3D &Object3D::translateOnAxis(const Vector3 &axis, float distance) {

    // translate object by distance along axis in object space
    // axis is assumed to be normalized

    _v1.copy(axis).applyQuaternion(this->quaternion);

    this->position.add(_v1.multiply(distance));

    return *this;
}

Object3D &Object3D::translateX(float distance) {

    return this->translateOnAxis(Vector3::X, distance);
}

Object3D &Object3D::translateY(float distance) {

    return this->translateOnAxis(Vector3::Y, distance);
}

Object3D &Object3D::translateZ(float distance) {

    return this->translateOnAxis(Vector3::Z, distance);
}

void Object3D::localToWorld(Vector3 &vector) const {

    vector.applyMatrix4(this->matrixWorld);
}

void Object3D::worldToLocal(Vector3 &vector) const {

    vector.applyMatrix4(_m1.copy(this->matrixWorld).invert());
}

void Object3D::lookAt(const Vector3 &vector) {

    lookAt(vector.x, vector.y, vector.z);
}

void Object3D::lookAt(float x, float y, float z) {

    // This method does not support objects having non-uniformly-scaled parent(s)

    _target.set(x, y, z);

    this->updateWorldMatrix(true, false);

    _position.setFromMatrixPosition(this->matrixWorld);

    if (instanceof <Camera>(this) || instanceof <Light>(this)) {

        _m1.lookAt(_position, _target, this->up);

    } else {

        _m1.lookAt(_target, _position, this->up);
    }

    this->quaternion.setFromRotationMatrix(_m1);

    if (parent) {

        _m1.extractRotation(parent->matrixWorld);
        _q1.setFromRotationMatrix(_m1);
        this->quaternion.premultiply(_q1.invert());
    }
}

Object3D &Object3D::add(Object3D *object) {

    if (object->parent) {

        object->parent->remove( object );
    }

    object->parent = this;
    this->children.emplace_back(object);

    object->dispatchEvent("added");

    return *this;
}

Object3D &Object3D::remove(Object3D *object) {

    auto find = std::find(children.begin(), children.end(), object);
    if (find != children.end()) {
        children.erase(find);
        object->parent = nullptr;
        object->dispatchEvent("remove");
    }

    return *this;
}

Object3D &Object3D::removeFromParent() {

    if (parent) {

        parent->remove(this);
    }

    return *this;
}

Object3D &Object3D::clear() {

    for (auto &object : this->children) {

        object->parent = nullptr;

        object->dispatchEvent("remove");
    }

    this->children.clear();

    return *this;
}

Object3D *Object3D::getObjectByName(const std::string &name) {

    if (this->name == name) return this;

    for (auto &child : this->children) {

        auto object = child->getObjectByName(name);

        if (object) {

            return object;
        }
    }

    return nullptr;
}

void Object3D::getWorldPosition(Vector3 &target) {

    this->updateWorldMatrix(true, false);

    target.setFromMatrixPosition(this->matrixWorld);
}

void Object3D::getWorldQuaternion(Quaternion &target) {

    this->updateWorldMatrix(true, false);

    this->matrixWorld.decompose(_position, target, _scale);
}

void Object3D::getWorldScale(Vector3 &target) {

    this->updateWorldMatrix(true, false);

    this->matrixWorld.decompose(_position, _quaternion, target);
}

void Object3D::getWorldDirection(Vector3 &target) {

    this->updateWorldMatrix(true, false);

    const auto& e = this->matrixWorld.elements;

    target.set(e[8], e[9], e[10]).normalize();
}

void Object3D::traverse(const std::function<void(Object3D &)> &callback) {

    callback(*this);

    for (auto &i : children) {

        i->traverse(callback);
    }
}

void Object3D::traverseVisible(const std::function<void(Object3D &)> &callback) {

    if (!this->visible) return;

    callback(*this);

    for (auto &i : children) {

        i->traverseVisible(callback);
    }
}

void Object3D::traverseAncestors(const std::function<void(Object3D &)> &callback) const {

    if (parent) {

        callback(*parent);

        parent->traverseAncestors(callback);
    }
}

void Object3D::updateMatrix() {

    this->matrix.compose(this->position, this->quaternion, this->scale);

    this->matrixWorldNeedsUpdate = true;
}

void Object3D::updateMatrixWorld(bool force) {

    if (this->matrixAutoUpdate) this->updateMatrix();

    if (this->matrixWorldNeedsUpdate || force) {

        if (!this->parent) {

            this->matrixWorld = (this->matrix);

        } else {

            this->matrixWorld.multiplyMatrices(this->parent->matrixWorld, this->matrix);
        }

        this->matrixWorldNeedsUpdate = false;

        force = true;
    }

    // update children

    for (auto &child : this->children) {

        child->updateMatrixWorld(force);
    }
}

void Object3D::updateWorldMatrix(bool updateParents, bool updateChildren) {

    if (updateParents && parent) {

        parent->updateWorldMatrix(true, false);
    }

    if (this->matrixAutoUpdate) this->updateMatrix();

    if (!this->parent) {

        this->matrixWorld = (this->matrix);

    } else {

        this->matrixWorld.multiplyMatrices(this->parent->matrixWorld, this->matrix);
    }

    // update children

    if (updateChildren) {

        for (auto &child : children) {

            child->updateWorldMatrix(false, true);
        }
    }
}
