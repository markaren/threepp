// https://github.com/mrdoob/three.js/blob/r129/src/math/Quaternion.js

#ifndef THREEPP_QUATERNION_HPP
#define THREEPP_QUATERNION_HPP

#include <functional>

namespace threepp {

    class vector3;
    class matrix4;

    class quaternion {

    public:
        quaternion() = default;

        quaternion(float x, float y, float z, float w);

        [[nodiscard]] float x() const {
            return x_;
        }

        [[nodiscard]] float y() const {
            return y_;
        }

        [[nodiscard]] float z() const {
            return z_;
        }

        [[nodiscard]] float w() const {
            return w_;
        }

        quaternion &set(float x, float y, float z, float w);

        quaternion &setFromAxisAngle(const vector3 &axis, float angle);

        quaternion &setFromRotationMatrix( const matrix4 &m );

        [[nodiscard]] float angleTo( const quaternion &q ) const;

        quaternion &identity();

        quaternion &invert();

        quaternion &conjugate();

        [[nodiscard]] float dot(const quaternion &v) const;

        [[nodiscard]] float lengthSq() const;

        [[nodiscard]] float length() const;

        quaternion &normalize();

        quaternion &_onChange( std::function<void()> callback );

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
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x_;
            array[offset + 1] = this->y_;
            array[offset + 2] = this->z_;
            array[offset + 3] = this->w_;
        }

    private:
        float x_ = 0.0;
        float y_ = 0.0;
        float z_ = 0.0;
        float w_ = 1.0;

        std::function<void()> onChangeCallback_ = [] {};
    };

}// namespace threepp

#endif//THREEPP_QUATERNION_HPP
