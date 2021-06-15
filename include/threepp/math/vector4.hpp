/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector4.js
 */

#ifndef THREEPP_VECTOR3_HPP
#define THREEPP_VECTOR3_HPP

#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>

#include "matrix3.hpp"
#include "matrix4.hpp"
#include "math_utils.hpp"

namespace threepp::math {

    class vector4 {

    public:
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double w = 1.0;

        vector4() = default;

        vector4(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {};

        vector4 &set(double x, double y, double z, double w) {

            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;

            return *this;
        }

        vector3 &setScalar(double value) {

            this->x = value;
            this->y = value;
            this->z = value;
            this->w = value;

            return *this;
        }

        template<class ArrayLike>
        vector3 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x = array[offset + 0];
            this->y = array[offset + 1];
            this->z = array[offset + 2];
            this->w = array[offset + 2];

            return *this;

        }

        vector4 &applyMatrix4( const matrix4 &m ) {

                const auto x_ = this.x, y_ = this.y, z_ = this.z, w_ = this.w;
                const e = m.elements;

                this.x = e[ 0 ] * x_ + e[ 4 ] * y_ + e[ 8 ] * z_ + e[ 12 ] * w_;
                this.y = e[ 1 ] * x_ + e[ 5 ] * y_ + e[ 9 ] * z_ + e[ 13 ] * w_;
                this.z = e[ 2 ] * x_ + e[ 6 ] * y_ + e[ 10 ] * z_ + e[ 14 ] * w_;
                this.w = e[ 3 ] * x_ + e[ 7 ] * y_ + e[ 11 ] * z_ + e[ 15 ] * w_;

                return *this;

        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) {

            array[offset + 0] = this->x;
            array[offset + 1] = this->y;
            array[offset + 2] = this->z;
            array[offset + 3] = this->w;

        }

    };

    std::ostream &operator<<(std::ostream &os, const vector4 &v) {
        return os << "vector3(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ", z=" + std::to_string(v.z) +
                     ", w=" + std::to_string(w) + + ")";
    }


}

#endif //THREEPP_VECTOR3_HPP
