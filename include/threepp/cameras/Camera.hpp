// https://github.com/mrdoob/three.js/blob/r129/src/cameras/Camera.js

#ifndef THREEPP_CAMERA_HPP
#define THREEPP_CAMERA_HPP

#include "threepp/cameras/View.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Matrix4.hpp"

namespace threepp {

    class Camera : public Object3D {

    public:

        int zoom = 1;

        float near;
        float far;

        std::optional<View> view;

        Camera(const Camera &) = delete;

        Matrix4 matrixWorldInverse = Matrix4();

        Matrix4 projectionMatrix = Matrix4();
        Matrix4 projectionMatrixInverse = Matrix4();

        std::string type() const override {
            return "Camera";
        }

        virtual void updateProjectionMatrix() = 0;

    protected:
        Camera(float near, float far): near(near), far(far) {};
    };

}// namespace threepp

#endif//THREEPP_CAMERA_HPP
