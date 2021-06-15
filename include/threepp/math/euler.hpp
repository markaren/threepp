// https://github.com/mrdoob/three.js/blob/r129/src/math/Euler.js

#ifndef THREEPP_EULER_HPP
#define THREEPP_EULER_HPP

#include <functional>
#include <optional>

namespace threepp {

    class vector3;
    class matrix3;
    class matrix4;
    class quaternion;

    class euler {

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

        euler() = default;

        [[nodiscard]] double x() const {
            return x_;
        }

        [[nodiscard]] double y() const {
            return y_;
        }

        [[nodiscard]] double z() const {
            return z_;
        }

        [[nodiscard]] RotationOrders order() const {
            return order_;
        }

        euler &set(double x, double y, double z, const std::optional<RotationOrders> &order = std::nullopt);

        euler &setFromRotationMatrix(const matrix4 &m, std::optional<RotationOrders> order = std::nullopt, bool update = true);

        euler &setFromQuaternion(const quaternion &q, std::optional<RotationOrders> order = std::nullopt, bool update = true);

        euler &setFromVector3(const vector3 &v, std::optional<RotationOrders> order = std::nullopt);

    private:
        double x_ = 0.0;
        double y_ = 0.0;
        double z_ = 0.0;
        RotationOrders order_ = default_order;

        std::function<void()> onChangeCallback_ = [] {};

        static quaternion _quaternion;
        static matrix4 _matrix;
    };


}// namespace threepp

#endif//THREEPP_EULER_HPP
