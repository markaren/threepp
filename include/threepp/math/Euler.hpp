// https://github.com/mrdoob/three.js/blob/r129/src/math/Euler.js

#ifndef THREEPP_EULER_HPP
#define THREEPP_EULER_HPP

#include "float_view.hpp"

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

        float_view x;
        float_view y;
        float_view z;

        Euler() = default;

        [[nodiscard]] RotationOrders getOrder() const {

            return order_;
        }

        void setOrder(RotationOrders value) {

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

            this->x.value = array[offset];
            this->y.value = array[offset + 1];
            this->z.value = array[offset + 2];

            this->onChangeCallback_();

            return *this;
        }

        template<class ArrayLike>
        void toArray(ArrayLike &array, unsigned int offset = 0) const {

            array[offset] = this->x();
            array[offset + 1] = this->y();
            array[offset + 2] = this->z();
        }


    private:
        RotationOrders order_ = default_order;

        std::function<void()> onChangeCallback_ = [] {};

        friend class Quaternion;
    };


}// namespace threepp

#endif//THREEPP_EULER_HPP
