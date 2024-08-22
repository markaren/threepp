
#include "threepp/controls/DragControls.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/objects/Group.hpp"

using namespace threepp;


struct DragControls::Impl: public MouseListener {

    Impl(DragControls& scope, const std::vector<Object3D*>& objects, Camera& camera, PeripheralsEventSource& eventSource)
        : scope(&scope), _camera(&camera), eventSource(&eventSource), _objects(objects) {

        activate();
    }

    void onMouseMove(const Vector2& pos) override {
        onPointerMove(pos);
    }

    void onMouseDown(int button, const Vector2& pos) override {
        if (button == 0) {
            onPointerDown(pos);
        }
    }

    void onMouseUp(int button, const Vector2&) override {
        if (button == 0) {
            onPointerCancel();
        }
    }

    void onPointerMove(Vector2 pos) {

        if (scope->enabled == false) return;

        updatePointer(pos);

        _raycaster.setFromCamera(_pointer, *_camera);

        if (_selected) {

            if (scope->mode == Mode::Translate) {

                _raycaster.ray.intersectPlane(_plane, _intersection);
                if (!_intersection.isNan()) {

                    _selected->position.copy(_intersection.sub(_offset).applyMatrix4(_inverseMatrix));
                }

            } else {

                _diff.subVectors(_pointer, _previousPointer).multiplyScalar(scope->rotateSpeed);
                _selected->rotateOnWorldAxis(_up, _diff.x);
                _selected->rotateOnWorldAxis(_right.normalize(), -_diff.y);
            }

            scope->dispatchEvent("drag", _selected);

            _previousPointer.copy(_pointer);

        } else {

            // hover support

            _raycaster.setFromCamera(_pointer, *_camera);
            _intersections = _raycaster.intersectObjects(_objects, scope->recursive);

            if (!_intersections.empty()) {

                const auto object = _intersections[0].object;

                Vector3 worldDir = _plane.normal;
                _camera->getWorldDirection(worldDir);
                _plane.setFromNormalAndCoplanarPoint(worldDir, _worldPosition.setFromMatrixPosition(*object->matrixWorld));

                if (_hovered && _hovered != object) {

                    scope->dispatchEvent("hoveroff", _hovered);

                    _hovered = nullptr;
                }

                if (_hovered != object) {

                    scope->dispatchEvent("hoveron", object);

                    _hovered = object;
                }

            } else {

                if (_hovered) {

                    scope->dispatchEvent("hoveroff", _hovered);

                    _hovered = nullptr;
                }
            }
        }

        _previousPointer.copy(_pointer);
    }

    void onPointerDown(Vector2 pos) {

        if (scope->enabled == false) return;

        updatePointer(pos);

        _raycaster.setFromCamera(_pointer, *_camera);
        _intersections = _raycaster.intersectObjects(_objects, scope->recursive);

        if (!_intersections.empty()) {

            if (scope->transformGroup == true) {

                // look for the outermost group in the object's upper hierarchy

                _selected = findGroup(_intersections[0].object);

            } else {

                _selected = _intersections[0].object;
            }

            Vector3 worldDir = _plane.normal;
            _camera->getWorldDirection(worldDir);
            _plane.setFromNormalAndCoplanarPoint(worldDir, _worldPosition.setFromMatrixPosition(*_selected->matrixWorld));

            _raycaster.ray.intersectPlane(_plane, _intersection);
            if (!_intersection.isNan()) {

                if (scope->mode == Mode::Translate) {

                    _inverseMatrix.copy(*_selected->parent->matrixWorld).invert();
                    _offset.copy(_intersection).sub(_worldPosition.setFromMatrixPosition(*_selected->matrixWorld));

                } else {

                    // the controls only support Y+ up
                    _up.set(0, 1, 0).applyQuaternion(_camera->quaternion).normalize();
                    _right.set(1, 0, 0).applyQuaternion(_camera->quaternion).normalize();
                }
            }

            scope->dispatchEvent("dragstart", _selected);
        }

        _previousPointer.copy(_pointer);
    }

    void onPointerCancel() {

        if (scope->enabled == false) return;

        if (_selected) {

            scope->dispatchEvent("dragend", _selected);

            _selected = nullptr;
        }
    }

    void updatePointer(Vector2 pos) {

        const auto rect = eventSource->size();

        _pointer.x = (pos.x / static_cast<float>(rect.width())) * 2 - 1;
        _pointer.y = -(pos.y / static_cast<float>(rect.height())) * 2 + 1;
    }

    Object3D* findGroup(Object3D* obj, Object3D* group = nullptr) {

        if (obj->is<Group>()) group = obj;

        if (!obj->parent) return group;

        return findGroup(obj->parent, group);
    }

    void activate() {

        eventSource->addMouseListener(*this);
    }

    void deactivate() {

        eventSource->removeMouseListener(*this);
    }

    ~Impl() override {
        deactivate();
    }

private:
    DragControls* scope;
    Camera* _camera;
    PeripheralsEventSource* eventSource;

    std::vector<Object3D*> _objects;
    Object3D* _selected = nullptr;
    Object3D* _hovered = nullptr;

    std::vector<Intersection> _intersections;

    // Mode mode{Mode::Translate};

    Plane _plane;
    Raycaster _raycaster;

    Vector2 _pointer;
    Vector3 _offset;
    Vector2 _diff;
    Vector2 _previousPointer;
    Vector3 _intersection;
    Vector3 _worldPosition;
    Matrix4 _inverseMatrix;

    Vector3 _up;
    Vector3 _right;

    friend class DragControls;
};


DragControls::DragControls(const std::vector<Object3D*>& objects, Camera& camera, PeripheralsEventSource& eventSource)
    : pimpl_(std::make_unique<Impl>(*this, objects, camera, eventSource)) {}

void DragControls::setObjects(const std::vector<Object3D*>& objects) {

    pimpl_->_objects = objects;
}

DragControls::~DragControls() = default;
