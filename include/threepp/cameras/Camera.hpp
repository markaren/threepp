// https://github.com/mrdoob/three.js/blob/r129/src/cameras/Camera.js

#ifndef THREEPP_CAMERA_HPP
#define THREEPP_CAMERA_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Matrix4.hpp"

namespace threepp {

    struct CameraView {

        bool enabled{};
        int fullWidth{};
        int fullHeight{};
        int offsetX{};
        int offsetY{};
        int width{};
        int height{};
    };

    // Abstract base class for cameras. This class should always be inherited when you build a new camera.
    class Camera: public Object3D {

    public:
        float zoom = 1;

        float near{};
        float far{};

        std::optional<CameraView> view;

        // This is the inverse of matrixWorld. MatrixWorld contains the Matrix which has the world transform of the Camera.
        Matrix4 matrixWorldInverse;

        // This is the matrix which contains the projection.
        Matrix4 projectionMatrix;
        // The inverse of projectionMatrix.
        Matrix4 projectionMatrixInverse;

        Camera() = default;
        Camera(float near, float far);
        Camera(const Camera&) = delete;

        // Copies the world space direction in which the camera is looking into target.
        // (Note: A camera looks down its local, negative z-axis).
        void getWorldDirection(Vector3& target) override;

        void updateMatrixWorld(bool force = false) override;

        void updateWorldMatrix(std::optional<bool> updateParents, std::optional<bool> updateChildren) override;

        virtual void updateProjectionMatrix(){};
    };

}// namespace threepp

#endif//THREEPP_CAMERA_HPP
