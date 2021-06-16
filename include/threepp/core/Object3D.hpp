// https://github.com/mrdoob/three.js/blob/r129/src/core/Object3D.js

#ifndef THREEPP_OBJECT3D_HPP
#define THREEPP_OBJECT3D_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/MathUtils.hpp"

#include <optional>

namespace threepp {

class Object3D {

public:

    const unsigned int id = _object3Did++;

    std::string uuid = generateUUID();

    std::string name;
    std::string type = "Object3D";

    std::shared_ptr<Object3D> parent;
    std::vector<std::shared_ptr<Object3D>> children;

    Vector3 position = Vector3();
    Euler rotation = Euler();
    Quaternion quaternion = Quaternion();
    Vector3 scale = Vector3(1,1,1);

    Matrix4 matrix = Matrix4();
    Matrix4 matrixWorld = Matrix4();

    bool matrixAutoUpdate = true;
    bool matrixWorldNeedsUpdate = false;

    bool visible = true;

    bool castShadows = true;
    bool receiveShadow = true;

    bool frustumCulled = true;
    unsigned int renderOrder = 0;

    Object3D() = default;
    
    void applyMatrix4( const Matrix4 &matrix ) {

        if ( this->matrixAutoUpdate ) this->updateMatrix();

        this->matrix.premultiply( matrix );

        this->matrix.decompose( this->position, this->quaternion, this->scale );

    }

    Object3D &applyQuaternion( const Quaternion &q ) {

            this->quaternion.premultiply( q );

            return *this;

    }

    void setRotationFromAxisAngle( const Vector3 &axis, float angle ) {

        // assumes axis is normalized

        this->quaternion.setFromAxisAngle( axis, angle );

    }

    void setRotationFromEuler( const Euler &euler ) {

            this->quaternion.setFromEuler( euler, true );

    }

    void setRotationFromMatrix( const Matrix4 &m ) {

            // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

            this->quaternion.setFromRotationMatrix( m );

    }

    void setRotationFromQuaternion( const Quaternion &q ) {

            // assumes q is normalized

            this->quaternion = q;

    }

    Object3D &rotateOnAxis( const Vector3 &axis, float angle ) {

        // rotate object on axis in object space
        // axis is assumed to be normalized

        _q1.setFromAxisAngle( axis, angle );

        this->quaternion.multiply( _q1 );

        return *this;

    }

    void updateMatrix() {

        this->matrix.compose( this->position, this->quaternion, this->scale );

        this->matrixWorldNeedsUpdate = true;

    }


private:

    std::function<void()> onRotationChange = [&]{

    };

    std::function<void()> onQuaternionChange = [&]{
        rotation.setFromQuaternion(quaternion, std::nullopt, false);
    };

  static Vector3 _v1;
  static Quaternion _q1;

  static unsigned int _object3Did;

};

}

#endif // THREEPP_OBJECT3D_HPP
