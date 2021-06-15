// https://github.com/mrdoob/three.js/blob/r129/src/math/Quaternion.js

#ifndef THREEPP_QUATERNION_HPP
#define THREEPP_QUATERNION_HPP

#include "vector3.hpp"

#include <cmath>
#include <functional>

namespace threepp::math {

    class quaternion {

    public:

        quaternion() = default;

        quaternion(double x, double y, double z, double w) : x_(x), y_(y), z_(z), w_(w) {};

        quaternion &set(double x, double y, double z, double w) {

            this->x_ = x;
            this->y_ = y;
            this->z_ = z;
            this->w_ = w;

            this->onChangeCallback_();

            return *this;

        }

        quaternion &setFromAxisAngle(const vector3 &axis, double angle) {

            // http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm

            // assumes axis is normalized

            const auto halfAngle = angle / 2, s = std::sin(halfAngle);

            this->x_ = axis.x * s;
            this->y_ = axis.y * s;
            this->z_ = axis.z * s;
            this->w_ = std::cos(halfAngle);

            this->onChangeCallback_();

            return *this;

        }


        quaternion &identity() {

            return this->set(0, 0, 0, 1);

        }

        quaternion &invert() {

            // quaternion is assumed to have unit length

            return this->conjugate();

        }

        quaternion &conjugate() {

            this->x_ *= -1;
            this->y_ *= -1;
            this->z_ *= -1;

            this->onChangeCallback_();

            return *this;

        }

        [[nodiscard]] double dot(const quaternion &v) const {

            return this->x_ * v.x_ + this->y_ * v.y_ + this->z_ * v.z_ + this->w_ * v.w_;

        }

        [[nodiscard]] double lengthSq() const {

            return this->x_ * this->x_ + this->y_ * this->y_ + this->z_ * this->z_ + this->w_ * this->w_;

        }

        [[nodiscard]] double length() const {

            return std::sqrt(this->x_ * this->x_ + this->y_ * this->y_ + this->z_ * this->z_ + this->w_ * this->w_);

        }

        quaternion &normalize() {

            auto l = length();

            if (l == 0) {

                this->x_ = 0;
                this->y_ = 0;
                this->z_ = 0;
                this->w_ = 1;

            } else {

                l = 1.0 / l;

                this->x_ = this->x_ * l;
                this->y_ = this->y_ * l;
                this->z_ = this->z_ * l;
                this->w_ = this->w_ * l;

            }

            this->onChangeCallback_();

            return *this;

        }

        template<class ArrayLike>
        quaternion &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x_ = array[offset];
            this->y_ = array[offset + 1];
            this->z_ = array[offset + 2];
            this->w_ = array[offset + 3];

            this->onChangeCallback_();

            return *this;

        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) {

            array[offset] = this->x_;
            array[offset + 1] = this->y_;
            array[offset + 2] = this->z_;
            array[offset + 3] = this->w_;

        }

    private:
        double x_ = 0.0;
        double y_ = 0.0;
        double z_ = 0.0;
        double w_ = 1.0;

        std::function<void()> onChangeCallback_ = []{};

    };

}

#endif //THREEPP_QUATERNION_HPP
