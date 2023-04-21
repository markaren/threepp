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

    class Camera: public Object3D {

    public:
        float zoom = 1;

        float near{};
        float far{};

        std::optional<CameraView> view;

        Matrix4 matrixWorldInverse;

        Matrix4 projectionMatrix;
        Matrix4 projectionMatrixInverse;

        Camera() = default;
        Camera(float near, float far);
        Camera(const Camera&) = delete;

        void getWorldDirection(Vector3& target) override;

        void updateMatrixWorld(bool force = false) override;

        void updateWorldMatrix(std::optional<bool> updateParents, std::optional<bool> updateChildren) override;

        virtual void updateProjectionMatrix(){};
    };

}// namespace threepp

#endif//THREEPP_CAMERA_HPP
