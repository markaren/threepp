
#ifndef THREEPP_TRANSFORMCONTROLSPLANE_HPP
#define THREEPP_TRANSFORMCONTROLSPLANE_HPP

#include "threepp/objects/Mesh.hpp"

#include "TransformControlsState.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

using namespace threepp;


class TransformControlsPlane: public Mesh {

public:

    explicit TransformControlsPlane(State& state)
        : state(state), Mesh(PlaneGeometry::create(100000, 100000, 2, 2),
                             MeshBasicMaterial::create(MeshBasicMaterial::Params{}.visible(false).wireframe(true).side(Side::Double).transparent(true).opacity(0.1f).toneMapped(false))) {}

    void updateMatrixWorld(bool force) override {

        auto space = state.space;

        this->position.copy(state.worldPosition);

        if (state.mode == "scale") space = "local";// scale always oriented to local rotation

        _v1.copy(_unitX).applyQuaternion(space == "local" ? state.worldQuaternion : _identityQuaternion);
        _v2.copy(_unitY).applyQuaternion(space == "local" ? state.worldQuaternion : _identityQuaternion);
        _v3.copy(_unitZ).applyQuaternion(space == "local" ? state.worldQuaternion : _identityQuaternion);

        // Align the plane for current transform mode, axis and space.

        _alignVector.copy(_v2);

        if (state.mode == "translate" || state.mode == "scale") {

            if (state.axis == "X") {

                _alignVector.copy(state.eye).cross(_v1);
                _dirVector.copy(_v1).cross(_alignVector);
            } else if (state.axis == "Y") {
                _alignVector.copy(state.eye).cross(_v2);
                _dirVector.copy(_v2).cross(_alignVector);
            } else if (state.axis == "Z") {
                _alignVector.copy(state.eye).cross(_v3);
                _dirVector.copy(_v3).cross(_alignVector);
            } else if (state.axis == "XY") {
                _dirVector.copy(_v3);
            } else if (state.axis == "YZ") {
                _dirVector.copy(_v1);
            } else if (state.axis == "XZ") {
                _alignVector.copy(_v3);
                _dirVector.copy(_v2);
            } else if (state.axis == "XYZ" || state.axis == "E") {

                _dirVector.set(0, 0, 0);
            }
        } else {

            _dirVector.set(0, 0, 0);
        }

        if (_dirVector.length() == 0) {

            // If in rotate mode, make the plane parallel to camera
            this->quaternion.copy(state.cameraQuaternion);

        } else {

            _tempMatrix.lookAt(_tempVector.set(0, 0, 0), _dirVector, _alignVector);

            this->quaternion.setFromRotationMatrix(_tempMatrix);
        }

        Object3D::updateMatrixWorld(force);
    }

private:

    State& state;

    Vector3 _tempVector;
    Quaternion _identityQuaternion;
    Matrix4 _tempMatrix;

    Vector3 _v1, _v2, _v3;
    Vector3 _unitX = Vector3(1, 0, 0), _unitY = Vector3(0, 1, 0), _unitZ = Vector3(0, 0, 1);
    Vector3 _dirVector, _alignVector;
};

#endif //THREEPP_TRANSFORMCONTROLSPLANE_HPP
