
#include "threepp/controls/TransformControls.hpp"

#include "threepp/cameras/Camera.hpp"

#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"

#include "threepp/core/Raycaster.hpp"

#include "threepp/input/PeripheralsEventSource.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "TransformControlsGizmo.hpp"
#include "TransformControlsPlane.hpp"


using namespace threepp;

namespace {

    std::unordered_map<std::string, Vector3> _unit{
            {"X", Vector3{1, 0, 0}},
            {"Y", Vector3{0, 1, 0}},
            {"Z", Vector3{0, 0, 1}}};

}// namespace


struct TransformControls::Impl {

    Raycaster _raycaster;

    Vector3 _tempVector;
    Vector3 _tempVector2;

    Quaternion _tempQuaternion;

    Vector3 _offset;
    Vector3 _startNorm;
    Vector3 _endNorm;
    Vector3 _cameraScale;

    Vector3 _parentPosition;
    Quaternion _parentQuaternion;
    Quaternion _parentQuaternionInv;
    Vector3 _parentScale;

    Vector3 _worldScaleStart;
    Quaternion _worldQuaternionInv;
    Vector3 _worldScale;

    Vector3 _positionStart;
    Quaternion _quaternionStart;
    Vector3 _scaleStart;

    Vector3 pointStart;
    Vector3 pointEnd;

    float rotationAngle{};

    TransformControls& scope;
    PeripheralsEventSource& canvas;
    Object3D* object = nullptr;

    State state;

    std::shared_ptr<TransformControlsGizmo> _gizmo;
    std::shared_ptr<TransformControlsPlane> _plane;

    struct MyMouseListener: MouseListener {

        Impl& scope;
        bool moveEnabled = false;

        explicit MyMouseListener(Impl& scope): scope(scope) {}

        void onMouseDown(int button, const Vector2& pos) override {

            if (!scope.state.enabled) return;

            button_ = button;
            moveEnabled = true;

            const auto rect = scope.canvas.size();

            Vector2 _pos;
            _pos.x = (pos.x / static_cast<float>(rect.width())) * 2.f - 1.f;
            _pos.y = -(pos.y / static_cast<float>(rect.height())) * 2.f + 1.f;

            // clamp to valid NDC range
            _pos.x = std::max(-1.f, std::min(1.f, _pos.x));
            _pos.y = std::max(-1.f, std::min(1.f, _pos.y));

            scope.pointerHover(_pos);
            scope.pointerDown(button, _pos);
        }

        void onMouseMove(const Vector2& pos) override {
            if (!scope.state.enabled) return;

            const auto rect = scope.canvas.size();

            Vector2 _pos;
            _pos.x = (pos.x / static_cast<float>(rect.width())) * 2.f - 1.f;
            _pos.y = -(pos.y / static_cast<float>(rect.height())) * 2.f + 1.f;

            _pos.x = std::max(-1.f, std::min(1.f, _pos.x));
            _pos.y = std::max(-1.f, std::min(1.f, _pos.y));

            scope.pointerHover(_pos);

            if (moveEnabled) {
                scope.pointerMove(button_, _pos);
            }
        }

        void onMouseUp(int button, const Vector2& pos) override {
            if (!scope.state.enabled) return;

            button_ = -1;
            moveEnabled = false;

            const auto rect = scope.canvas.size();

            Vector2 _pos;
            _pos.x = (pos.x / static_cast<float>(rect.width())) * 2.f - 1.f;
            _pos.y = -(pos.y / static_cast<float>(rect.height())) * 2.f + 1.f;

            _pos.x = std::max(-1.f, std::min(1.f, _pos.x));
            _pos.y = std::max(-1.f, std::min(1.f, _pos.y));

            scope.pointerUp(button, _pos);
        }


    private:
        int button_{-1};
    };


    MyMouseListener myMouseListener;

    Impl(TransformControls& scope, Camera& camera, PeripheralsEventSource& canvas)
        : scope(scope), canvas(canvas),
          state(State(scope.enabled, scope.showX, scope.showY, scope.showZ)),
          _gizmo(std::make_shared<TransformControlsGizmo>(state)),
          _plane(std::make_shared<TransformControlsPlane>(state)),
          myMouseListener(*this) {

        camera.updateMatrixWorld();
        camera.matrixWorld->decompose(this->state.cameraPosition, state.cameraQuaternion, this->_cameraScale);

        state.eye.copy(this->state.cameraPosition).sub(state.worldPosition).normalize();
        state.camera = &camera;

        canvas.addMouseListener(myMouseListener);

        _raycaster.params.lineThreshold = 0.1f;
    }

    ~Impl() {
        canvas.removeMouseListener(myMouseListener);
    }

    static std::optional<Intersection> intersectObjectWithRay(Object3D& object, Raycaster& raycaster, bool includeInvisible = false) {

        const auto allIntersections = raycaster.intersectObject(object, true);

        for (const auto& allIntersection : allIntersections) {

            if (allIntersection.object->visible || includeInvisible) {

                return allIntersection;
            }
        }

        return std::nullopt;
    }

    void pointerHover(const Vector2& pointer) {

        if (!this->object || state.dragging) return;

        _raycaster.setFromCamera(pointer, *this->state.camera);

        const auto intersect = intersectObjectWithRay(*this->_gizmo->picker[state.mode], _raycaster);

        if (intersect) {

            this->state.axis = intersect->object->name;

        } else {

            this->state.axis = std::nullopt;
        }
    }

    void pointerDown(int button, Vector2 pointer) {

        if (!this->object || state.dragging || button != 0) return;

        if (this->state.axis) {

            _raycaster.setFromCamera(pointer, *this->state.camera);

            const auto planeIntersect = intersectObjectWithRay(*this->_plane, _raycaster, true);

            if (planeIntersect) {

                auto space = state.space;

                if (state.mode == "scale") {

                    space = "local";

                } else if (this->state.axis == "E" || this->state.axis == "XYZE" || this->state.axis == "XYZ") {

                    space = "world";
                }

                if (space == "local" && state.mode == "rotate") {

                    const auto snap = state.rotationSnap;

                    if (this->state.axis == "X" && snap) this->object->rotation.x = std::round(this->object->rotation.x / *snap) * *snap;
                    if (this->state.axis == "Y" && snap) this->object->rotation.y = std::round(this->object->rotation.y / *snap) * *snap;
                    if (this->state.axis == "Z" && snap) this->object->rotation.z = std::round(this->object->rotation.z / *snap) * *snap;
                }

                this->object->updateMatrixWorld();
                this->object->parent->updateMatrixWorld();

                this->_positionStart.copy(this->object->position);
                this->_quaternionStart.copy(this->object->quaternion);
                this->_scaleStart.copy(this->object->scale);

                this->object->matrixWorld->decompose(this->state.worldPositionStart, this->state.worldQuaternionStart, this->_worldScaleStart);

                this->pointStart.copy(planeIntersect->point).sub(this->state.worldPositionStart);
            }

            state.dragging = true;
            scope.dispatchEvent("dragging-changed", this->state.dragging);
            scope.dispatchEvent("mouseDown", this->state.mode);
        }
    }

    void pointerMove(int button, Vector2 pointer) {

        const auto axis = this->state.axis;
        const auto mode = this->state.mode;
        const auto object = this->object;
        auto space = this->state.space;

        if (mode == "scale") {

            space = "local";

        } else if (axis == "E" || axis == "XYZE" || axis == "XYZ") {

            space = "world";
        }

        if (!object || !axis || this->state.dragging == false) return;

        _raycaster.setFromCamera(pointer, *this->state.camera);

        const auto planeIntersect = intersectObjectWithRay(*this->_plane, _raycaster, true);

        if (!planeIntersect) return;

        this->pointEnd.copy(planeIntersect->point).sub(this->state.worldPositionStart);

        if (mode == "translate") {

            // Apply translate

            this->_offset.copy(this->pointEnd).sub(this->pointStart);

            if (space == "local" && axis != "XYZ") {

                this->_offset.applyQuaternion(this->_worldQuaternionInv);
            }

            if (axis->find('X') == std::string::npos) this->_offset.x = 0;
            if (axis->find('Y') == std::string::npos) this->_offset.y = 0;
            if (axis->find('Z') == std::string::npos) this->_offset.z = 0;

            if (space == "local" && axis != "XYZ") {

                this->_offset.applyQuaternion(this->_quaternionStart).divide(this->_parentScale);

            } else {

                this->_offset.applyQuaternion(this->_parentQuaternionInv).divide(this->_parentScale);
            }

            object->position.copy(this->_offset).add(this->_positionStart);

            // Apply translation snap

            if (this->state.translationSnap) {

                const float snap = this->state.translationSnap.value();

                if (space == "local") {

                    object->position.applyQuaternion(_tempQuaternion.copy(this->_quaternionStart).invert());

                    if (axis->find('X') != std::string::npos) {

                        object->position.x = std::round(object->position.x / snap) * snap;
                    }

                    if (axis->find('Y') != std::string::npos) {

                        object->position.y = std::round(object->position.y / snap) * snap;
                    }

                    if (axis->find('Z') != std::string::npos) {

                        object->position.z = std::round(object->position.z / snap) * snap;
                    }

                    object->position.applyQuaternion(this->_quaternionStart);
                }

                if (space == "world") {

                    if (object->parent) {

                        object->position.add(_tempVector.setFromMatrixPosition(*object->parent->matrixWorld));
                    }

                    if (axis->find('X') != std::string::npos) {

                        object->position.x = std::round(object->position.x / snap) * snap;
                    }

                    if (axis->find('Y') != std::string::npos) {

                        object->position.y = std::round(object->position.y / snap) * snap;
                    }

                    if (axis->find('Z') != std::string::npos) {

                        object->position.z = std::round(object->position.z / snap) * snap;
                    }

                    if (object->parent) {

                        object->position.sub(_tempVector.setFromMatrixPosition(*object->parent->matrixWorld));
                    }
                }
            }

        } else if (mode == "scale") {

            if (axis->find("XYZ") != std::string::npos) {

                auto d = this->pointEnd.length() / this->pointStart.length();

                if (this->pointEnd.dot(this->pointStart) < 0) d *= -1;

                _tempVector2.set(d, d, d);

            } else {

                _tempVector.copy(this->pointStart);
                _tempVector2.copy(this->pointEnd);

                _tempVector.applyQuaternion(this->_worldQuaternionInv);
                _tempVector2.applyQuaternion(this->_worldQuaternionInv);

                _tempVector2.divide(_tempVector);

                if (axis->find('X') == std::string::npos) {

                    _tempVector2.x = 1;
                }

                if (axis->find('Y') == std::string::npos) {

                    _tempVector2.y = 1;
                }

                if (axis->find('Z') == std::string::npos) {

                    _tempVector2.z = 1;
                }
            }

            // Apply scale

            object->scale.copy(this->_scaleStart).multiply(_tempVector2);

            if (state.scaleSnap) {

                if (axis->find('X') != std::string::npos) {

                    auto snapped = std::round(object->scale.x / state.scaleSnap.value()) * state.scaleSnap.value();
                    object->scale.x = (snapped != 0) ? snapped : *state.scaleSnap;
                }

                if (axis->find('Y') != std::string::npos) {

                    auto snapped = std::round(object->scale.y / state.scaleSnap.value()) * state.scaleSnap.value();
                    object->scale.y = (snapped != 0) ? snapped : *state.scaleSnap;
                }

                if (axis->find('Z') != std::string::npos) {

                    auto snapped = std::round(object->scale.z / state.scaleSnap.value()) * state.scaleSnap.value();
                    object->scale.z = (snapped != 0) ? snapped : *state.scaleSnap;
                }
            }

        } else if (mode == "rotate") {

            this->_offset.copy(this->pointEnd).sub(this->pointStart);

            const auto ROTATION_SPEED = 20.f / this->state.worldPosition.distanceTo(_tempVector.setFromMatrixPosition(*this->state.camera->matrixWorld));

            if (axis == "E") {

                this->state.rotationAxis.copy(this->state.eye);
                this->rotationAngle = this->pointEnd.angleTo(this->pointStart);

                this->_startNorm.copy(this->pointStart).normalize();
                this->_endNorm.copy(this->pointEnd).normalize();

                this->rotationAngle *= (this->_endNorm.cross(this->_startNorm).dot(state.eye) < 0 ? 1.f : -1.f);

            } else if (axis == "XYZE") {

                this->state.rotationAxis.copy(this->_offset).cross(state.eye).normalize();
                this->rotationAngle = this->_offset.dot(_tempVector.copy(this->state.rotationAxis).cross(state.eye)) * ROTATION_SPEED;

            } else if (axis == "X" || axis == "Y" || axis == "Z") {

                this->state.rotationAxis.copy(_unit[*axis]);

                _tempVector.copy(_unit[*axis]);

                if (space == "local") {

                    _tempVector.applyQuaternion(state.worldQuaternion);
                }

                this->rotationAngle = this->_offset.dot(_tempVector.cross(state.eye).normalize()) * ROTATION_SPEED;
            }

            // Apply rotation snap

            if (this->state.rotationSnap) this->rotationAngle = std::round(this->rotationAngle / *this->state.rotationSnap) * *this->state.rotationSnap;

            // Apply rotate
            if (space == "local" && axis != "E" && axis != "XYZE") {

                object->quaternion.copy(this->_quaternionStart);
                object->quaternion.multiply(_tempQuaternion.setFromAxisAngle(this->state.rotationAxis, this->rotationAngle)).normalize();

            } else {

                this->state.rotationAxis.applyQuaternion(this->_parentQuaternionInv);
                object->quaternion.copy(_tempQuaternion.setFromAxisAngle(this->state.rotationAxis, this->rotationAngle));
                object->quaternion.multiply(this->_quaternionStart).normalize();
            }
        }

        this->scope.dispatchEvent("change");
        this->scope.dispatchEvent("objectChange");
    }

    void pointerUp(int button, Vector2) {

        if (button != 0) return;

        if (this->state.dragging && this->state.axis) {

            this->scope.dispatchEvent("mouseUp", &this->state.mode);
        }

        this->state.dragging = false;
        this->state.axis = std::nullopt;

        this->scope.dispatchEvent("dragging-changed", this->state.dragging);
    }

    void attach(Object3D& object) {

        scope.visible = true;
        this->object = &object;
    }

    void detach() {

        this->object = nullptr;
        scope.visible = false;
        this->state.axis = std::nullopt;
    }
};

TransformControls::TransformControls(Camera& camera, PeripheralsEventSource& canvas)
    : pimpl_(std::make_unique<Impl>(*this, camera, canvas)) {

    this->visible = false;

    this->add(pimpl_->_gizmo);
    this->add(pimpl_->_plane);

    Object3D::updateMatrixWorld();
}

void TransformControls::setSpace(const std::string& space) {
    pimpl_->state.space = space;
}

std::string TransformControls::getSpace() const {
    return pimpl_->state.space;
}

void TransformControls::setMode(const std::string& mode) {
    pimpl_->state.mode = mode;
}

void TransformControls::setSize(float size) {
    pimpl_->state.size = size;
}

void TransformControls::setTranslationSnap(std::optional<float> snap) {
    pimpl_->state.translationSnap = snap;
}

void TransformControls::setRotationSnap(std::optional<float> snap) {
    pimpl_->state.rotationSnap = snap;
}

void TransformControls::setScaleSnap(std::optional<float> snap) {
    pimpl_->state.scaleSnap = snap;
}

void TransformControls::updateMatrixWorld(bool force) {

    if (pimpl_->object) {

        pimpl_->object->updateMatrixWorld();

        if (!pimpl_->object->parent) {

            std::cerr << "TransformControls: The attached 3D object must be a part of the scene graph." << std::endl;

        } else {

            pimpl_->object->parent->matrixWorld->decompose(pimpl_->_parentPosition, pimpl_->_parentQuaternion, pimpl_->_parentScale);
        }

        pimpl_->object->matrixWorld->decompose(pimpl_->state.worldPosition, pimpl_->state.worldQuaternion, pimpl_->_worldScale);

        pimpl_->_parentQuaternionInv.copy(pimpl_->_parentQuaternion).invert();
        pimpl_->_worldQuaternionInv.copy(pimpl_->state.worldQuaternion).invert();
    }

    pimpl_->state.camera->updateMatrixWorld();
    pimpl_->state.camera->matrixWorld->decompose(pimpl_->state.cameraPosition, pimpl_->state.cameraQuaternion, pimpl_->_cameraScale);

    pimpl_->state.eye.copy(pimpl_->state.cameraPosition).sub(pimpl_->state.worldPosition).normalize();

    Object3D::updateMatrixWorld(true);
}

TransformControls& TransformControls::attach(Object3D& object) {

    pimpl_->attach(object);

    return *this;
}

TransformControls& TransformControls::detach() {

    pimpl_->detach();

    return *this;
}

TransformControls::~TransformControls() = default;
