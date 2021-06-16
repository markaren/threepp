// https://github.com/mrdoob/three.js/blob/r129/src/cameras/Camera.js

#ifndef THREEPP_CAMERA_HPP
#define THREEPP_CAMERA_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Matrix4.hpp"

namespace threepp {

    class Camera: public Object3D {

    public:

        std::string type = "Camera";

        Matrix4 matrixWorldInverse = Matrix4();

        Matrix4 projectionMatrix = Matrix4();
        Matrix4 projectionMatrixInverse = Matrix4();

    };

}

#endif//THREEPP_CAMERA_HPP
