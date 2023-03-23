
#ifndef THREEPP_TRIANGLE_HPP
#define THREEPP_TRIANGLE_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"

namespace threepp {

    class Triangle {

    public:
        Triangle() = default;

        Triangle(Vector3 a, Vector3 b, Vector3 c);

        const Vector3& operator[](char c) const;

        static void getNormal(const Vector3& a, const Vector3& b, const Vector3& c, Vector3& target);

        // static/instance method to calculate barycentric coordinates
        // based on: http://www.blackpawn.com/texts/pointinpoly/default.html
        static void
        getBarycoord(const Vector3& point, const Vector3& a, const Vector3& b, const Vector3& c, Vector3& target);

        static bool containsPoint(const Vector3& point, const Vector3& a, const Vector3& b, const Vector3& c);

        static void
        getUV(const Vector3& point, const Vector3& p1, const Vector3& p2, const Vector3& p3, const Vector2& uv1,
              const Vector2& uv2, const Vector2& uv3, Vector2& target);

        static bool isFrontFacing(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& direction);

        [[nodiscard]] const Vector3& a() const;

        [[nodiscard]] const Vector3& b() const;

        [[nodiscard]] const Vector3& c() const;

        Triangle& set(const Vector3& a, const Vector3& b, const Vector3& c);

        template<class ArrayLike>
        Triangle& setFromPointsAndIndices(const ArrayLike& points, unsigned int i0, unsigned int i1, unsigned int i2) {

            this->a_.copy(points[i0]);
            this->b_.copy(points[i1]);
            this->c_.copy(points[i2]);

            return *this;
        }

        [[nodiscard]] float getArea() const;

        void getMidpoint(Vector3& target);

        void getNormal(Vector3& target);

        void getBarycoord(Vector3& point, Vector3& target);

        void getUV(const Vector3& point, const Vector2& uv1, const Vector2& uv2, const Vector2& uv3, Vector2& target);

        bool containsPoint(const Vector3& point);

        bool isFrontFacing(const Vector3& direction);

        void closestPointToPoint(const Vector3& p, Vector3& target);

    private:
        Vector3 a_{};
        Vector3 b_{};
        Vector3 c_{};
    };


}// namespace threepp

#endif//THREEPP_TRIANGLE_HPP
