/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector4.js
 */

#ifndef THREEPP_VECTOR4_HPP
#define THREEPP_VECTOR4_HPP

#include <string>

namespace threepp {

    class Matrix3;
    class Matrix4;

    class Vector4 {

    public:
        float x = 0.0;
        float y = 0.0;
        float z = 0.0;
        float w = 1.0;

        Vector4() = default;

        Vector4(float x, float y, float z, float w);
        ;

        Vector4 &set(float x, float y, float z, float w);

        Vector4 &setScalar(float value);

        Vector4 &applyMatrix4(const Matrix4 &m);

        template<class ArrayLike>
        Vector4 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x = array[offset + 0];
            this->y = array[offset + 1];
            this->z = array[offset + 2];
            this->w = array[offset + 2];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset + 0] = this->x;
            array[offset + 1] = this->y;
            array[offset + 2] = this->z;
            array[offset + 3] = this->w;
        }
    };

    //    std::ostream &operator<<(std::ostream &os, const Vector4 &v) {
    //        return os << "Vector4(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
    //                     ", w=" + std::to_string(v.w) + +")";
    //    }


}// namespace threepp

#endif//THREEPP_VECTOR4_HPP
