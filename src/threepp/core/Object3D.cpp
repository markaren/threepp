
#include "threepp/core/Object3D.hpp"

using namespace threepp;

Vector3 Object3D::_v1 = Vector3();
Quaternion Object3D::_q1 = Quaternion();

Vector3 Object3D::_scale = Vector3();
Vector3 Object3D::_position = Vector3();

Quaternion Object3D::_quaternion = Quaternion();

unsigned int Object3D::_object3Did = 0;
