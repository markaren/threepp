/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector2.js
 */

#ifndef THREEPP_VECTOR2_HPP
#define THREEPP_VECTOR2_HPP

#include <string>

namespace threepp {

    class vector2 {

    public:
        double x = 0.0;
        double y = 0.0;

        vector2() = default;

        vector2(double x, double y);
        ;

        vector2 &set(double x, double y);

        vector2 &setScalar(double value);

        vector2 &setX(double value);

        vector2 &setY(double value);

        double &operator[](unsigned int index);

        vector2 &add(const vector2 &v);

        vector2 &add(double s);

        vector2 &addVectors(const vector2 &a, const vector2 &b);

        vector2 &addScaledVector(const vector2 &v, double s);

        template<class ArrayLike>
        vector2 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x = array[offset];
            this->y = array[offset + 1];

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x;
            array[offset + 1] = this->y;
        }
    };

    //    std::ostream &operator<<(std::ostream &os, const vector2 &v) {
    //        return os << "vector2(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ")";
    //    }

}// namespace threepp

#endif//THREEPP_VECTOR3_HPP
