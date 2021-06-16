// https://github.com/mrdoob/three.js/blob/r129/src/math/Euler.js

#ifndef THREEPP_EULER_HPP
#define THREEPP_EULER_HPP

#include <functional>
#include <optional>

namespace threepp {

    class Vector3;
    class Matrix3;
    class Matrix4;
    class Quaternion;

    class Euler {

    public:
        enum RotationOrders {
            XYZ,
            YZX,
            ZXY,
            XZY,
            YXZ,
            ZYX
        };

        const static RotationOrders default_order = XYZ;

        Euler() = default;

        [[nodiscard]] float x() const {
            return x_;
        }

        void x(float value) {

            this->x_ = value;
            onChangeCallback_();
        }

        [[nodiscard]] float y() const {
            return y_;
        }

        void y(float value) {

            this->y_ = value;
            onChangeCallback_();
        }

        [[nodiscard]] float z() const {
            return z_;
        }

        void z(float value) {

            this->z_ = value;
            onChangeCallback_();
        }

        [[nodiscard]] RotationOrders order() const {
            return order_;
        }

        void order(RotationOrders value) {

            this->order_ = value;
            onChangeCallback_();
        }

        Euler &set(float x, float y, float z, const std::optional<RotationOrders> &order = std::nullopt);

        Euler &setFromRotationMatrix(const Matrix4 &m, std::optional<RotationOrders> order = std::nullopt, bool update = true);

        Euler &setFromQuaternion(const Quaternion &q, std::optional<RotationOrders> order = std::nullopt, bool update = true);

        Euler &setFromVector3(const Vector3 &v, std::optional<RotationOrders> order = std::nullopt);

        Euler &_onChange(std::function<void()> callback);

        template<class ArrayLike>
        Euler &fromArray(const ArrayLike &array, unsigned int offset = 0) {

            this->x_ = array[offset];
            this->y_ = array[offset + 1];
            this->z_ = array[offset + 2];

            this->onChangeCallback_();

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x_;
            array[offset + 1] = this->y_;
            array[offset + 2] = this->z_;
        }


    private:
        float x_ = 0.0;
        float y_ = 0.0;
        float z_ = 0.0;
        RotationOrders order_ = default_order;

        std::function<void()> onChangeCallback_ = [] {};

        static Quaternion _quaternion;
        static Matrix4 _matrix;

        friend class Quaternion;
    };


}// namespace threepp

#endif//THREEPP_EULER_HPP
