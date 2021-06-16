// https://github.com/mrdoob/three.js/blob/r129/src/cameras/Camera.js

#ifndef THREEPP_CAMERA_HPP
#define THREEPP_CAMERA_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/core/Object3d.hpp"

namespace threepp {

    class Camera: public Object3d {

    public:

        virtual Camera() = 0;

        std::string type = "Camera";

        Matrix4 matrixWorldInverse = Matrix4();

        Matrix4 projectionMatrix = Matrix4();
        Matrix4 projectionMatrixInverse = Matrix4();

    };

}

#endif//THREEPP_CAMERA_HPP
