/*
 * https://github.com/mrdoob/three.js/blob/r129/src/math/Vector2.js
 */

#ifndef THREEPP_VECTOR2_HPP
#define THREEPP_VECTOR2_HPP

#include <cmath>
#include <algorithm>
#include <string>
#include <stdexcept>

#include "matrix3.hpp"
#include "matrix4.hpp"
#include "math_utils.hpp"

namespace threepp::math {

    class vector2 {

    public:
        double x = 0.0;
        double y = 0.0;

        vector2() = default;

        vector2(double x, double y) : x(x), y(y) {};

        vector2 &set(double x, double y) {

            this->x = x;
            this->y = y;

            return *this;
        }

        vector2 &setScalar(double value) {

            this->x = value;
            this->y = value;

            return *this;
        }

        vector2 &setX(double value) {

            this->x = value;

            return *this;

        }

        vector2 &setY(double value) {

            y = value;

            return *this;

        }

        vector2 &setComponent(unsigned int index, double value) {

            switch (index) {

                case 0:
                    x = value;
                    break;
                case 1:
                    y = value;
                    break;
                default:
                    throw std::runtime_error("index is out of range: " + std::to_string(index));

            }

            return *this;

        }

        [[nodiscard]] double getComponent(unsigned int index) const {

            switch (index) {

                case 0:
                    return x;
                case 1:
                    return y;
                default:
                    throw std::runtime_error("index is out of range: " + std::to_string(index));

            }

        }

        vector2 &add(const vector2 &v) {

            this->x += v.x;
            this->y += v.y;

            return *this;

        }

        vector2 &add(double s) {

            this->x += s;
            this->y += s;

            return *this;

        }

        vector2 &addVectors(const vector2 &a, const vector2 &b) {

            this->x = a.x + b.x;
            this->y = a.y + b.y;

            return *this;

        }

        vector2 &addScaledVector(const vector2 &v, double s) {

            this->x += v.x * s;
            this->y += v.y * s;

            return *this;

        }

        template<class ArrayLike>
        vector2 &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x = array[offset];
            this->y = array[offset + 1];

            return *this;

        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) {

            array[offset] = this->x;
            array[offset + 1] = this->y;

        }


    };

    std::ostream &operator<<(std::ostream &os, const vector2 &v) {
        return os << "vector3(x=" + std::to_string(v.x) + ", y=" + std::to_string(v.y) + ")";
    }


}

#endif //THREEPP_VECTOR3_HPP
