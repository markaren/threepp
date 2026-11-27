
#include "threepp/cameras/StereoCamera.hpp"

#include "threepp/math/MathUtils.hpp"

#include <cmath>

using namespace threepp;

namespace {

    // The eye's place in the world goes into `matrix`, not only matrixWorld:
    // with matrixAutoUpdate off, Object3D::updateMatrixWorld (which a renderer
    // calls on a parentless camera) copies matrix to matrixWorld whenever
    // matrix has changed, and on its first call.
    void place(PerspectiveCamera& eye, const Matrix4& cameraWorld, float xOffset) {

        Matrix4 offset;
        offset.elements[12] = xOffset;

        eye.matrix->multiplyMatrices(cameraWorld, offset);
        eye.matrixWorld->copy(*eye.matrix);
        eye.matrixWorldInverse.copy(*eye.matrixWorld).invert();
    }

}// namespace

StereoCamera::StereoCamera() {

    cameraL.layers.enable(1);
    cameraL.matrixAutoUpdate = false;

    cameraR.layers.enable(2);
    cameraR.matrixAutoUpdate = false;
}

void StereoCamera::update(const PerspectiveCamera& camera) {

    const Cache current{camera.focus, camera.fov, camera.aspect * aspect,
                        camera.nearPlane, camera.farPlane, camera.zoom, eyeSep};

    if (!cacheValid_ || !(current == cache_)) {

        cache_ = current;
        cacheValid_ = true;

        // Off-axis stereoscopic effect based on
        // http://paulbourke.net/stereographics/stereorender/

        Matrix4 projectionMatrix = camera.projectionMatrix;
        const float eyeSepOnProjection = (eyeSep / 2) * current.nearPlane / current.focus;
        const float ymax = (current.nearPlane * std::tan(math::DEG2RAD * current.fov * 0.5f)) / current.zoom;

        auto apply = [&](PerspectiveCamera& eye, float shift) {
            const float xmin = -ymax * current.aspect + shift;
            const float xmax = ymax * current.aspect + shift;

            projectionMatrix.elements[0] = 2 * current.nearPlane / (xmax - xmin);
            projectionMatrix.elements[8] = (xmax + xmin) / (xmax - xmin);

            eye.projectionMatrix.copy(projectionMatrix);
            eye.projectionMatrixInverse.copy(projectionMatrix).invert();

            eye.fov = current.fov;
            eye.aspect = current.aspect;
            eye.nearPlane = current.nearPlane;
            eye.farPlane = current.farPlane;
            eye.zoom = current.zoom;
            eye.focus = current.focus;
        };

        apply(cameraL, eyeSepOnProjection);
        apply(cameraR, -eyeSepOnProjection);
    }

    place(cameraL, *camera.matrixWorld, -eyeSep / 2);
    place(cameraR, *camera.matrixWorld, eyeSep / 2);
}
