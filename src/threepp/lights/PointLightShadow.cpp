
#include "threepp/lights/PointLightShadow.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/lights/PointLight.hpp"

using namespace threepp;


PointLightShadow::PointLightShadow()
    : LightShadow(std::make_unique<PerspectiveCamera>(90.f, 1.f, 0.5f, 500.f)),
      _cubeDirections({Vector3(1, 0, 0), Vector3(-1, 0, 0), Vector3(0, 0, 1),
                       Vector3(0, 0, -1), Vector3(0, 1, 0), Vector3(0, -1, 0)}),
      _cubeUps({Vector3(0, 1, 0), Vector3(0, 1, 0), Vector3(0, 1, 0),
                Vector3(0, 1, 0), Vector3(0, 0, 1), Vector3(0, 0, -1)}) {


    this->_frameExtents = Vector2(4, 2);

    _viewports = std::vector<Vector4>{// These viewports map a cube-map onto a 2D texture with the
                                      // following orientation:
                                      //
                                      //  xzXZ
                                      //   y Y
                                      //
                                      // X - Positive x direction
                                      // x - Negative x direction
                                      // Y - Positive y direction
                                      // y - Negative y direction
                                      // Z - Positive z direction
                                      // z - Negative z direction

                                      // positive X
                                      Vector4(2, 1, 1, 1),
                                      // negative X
                                      Vector4(0, 1, 1, 1),
                                      // positive Z
                                      Vector4(3, 1, 1, 1),
                                      // negative Z
                                      Vector4(1, 1, 1, 1),
                                      // positive Y
                                      Vector4(3, 0, 1, 1),
                                      // negative Y
                                      Vector4(1, 0, 1, 1)};
}

void PointLightShadow::updateMatrices(PointLight* light, size_t viewportIndex) {

    auto& camera = this->camera;
    auto& shadowMatrix = this->matrix;

    auto far = (light->distance > 0) ? light->distance : camera->far;

    if (far != camera->far) {

        camera->far = far;
        camera->updateProjectionMatrix();
    }

    _lightPositionWorld.setFromMatrixPosition(*light->matrixWorld);
    camera->position.copy(_lightPositionWorld);

    _lookTarget.copy(camera->position);
    _lookTarget.add(this->_cubeDirections[viewportIndex]);
    camera->up.copy(this->_cubeUps[viewportIndex]);
    camera->lookAt(_lookTarget);
    camera->updateMatrixWorld();

    shadowMatrix.makeTranslation(-_lightPositionWorld.x, -_lightPositionWorld.y, -_lightPositionWorld.z);

    _projScreenMatrix.multiplyMatrices(camera->projectionMatrix, camera->matrixWorldInverse);
    this->_frustum.setFromProjectionMatrix(_projScreenMatrix);
}

std::shared_ptr<PointLightShadow> PointLightShadow::create() {

    return std::shared_ptr<PointLightShadow>(new PointLightShadow());
}
