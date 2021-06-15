/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector4.js
 */

#ifndef THREEPP_VECTOR4_HPP
#define THREEPP_VECTOR4_HPP

#include <string>

namespace threepp {

    class matrix3;
    class matrix4;

    class vector4 {

    public:
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double w = 1.0;

        vector4() = default;

        vector4(double x, double y, double z, double w);
        ;

        vector4 &set(double x, double y, double z, double w);

        vector4 &setScalar(double value);

        vector4 &applyMatrix4(const matrix4 &m);

        template<class ArrayLike>
        vector4 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

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

    //    std::ostream &operator<<(std::ostream &os, const vector4 &v) {
    //        return os << "vector4(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
    //                     ", w=" + std::to_string(v.w) + +")";
    //    }


}// namespace threepp

#endif//THREEPP_VECTOR4_HPP
