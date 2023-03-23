// https://github.com/mrdoob/three.js/blob/r129/src/math/Plane.js

#ifndef THREEPP_PLANE_HPP
#define THREEPP_PLANE_HPP

#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Sphere;
    class Line3;
    class Box3;
    class Matrix3;

    class Plane {

    public:
        Vector3 normal;
        float constant;

        Plane();

        Plane(Vector3 normal, float constant);

        Plane& set(const Vector3& normal, float constant);

        Plane& setComponents(float x, float y, float z, float w);

        Plane& setFromNormalAndCoplanarPoint(const Vector3& normal, const Vector3& point);

        Plane& setFromCoplanarPoints(const Vector3& a, const Vector3& b, const Vector3& c);

        Plane& copy(const Plane& plane);

        Plane& normalize();

        Plane& negate();

        [[nodiscard]] float distanceToPoint(const Vector3& point) const;

        [[nodiscard]] float distanceToSphere(const Sphere& sphere) const;

        void projectPoint(const Vector3& point, Vector3& target) const;

        void intersectLine(const Line3& line, Vector3& target) const;

        [[nodiscard]] bool intersectsLine(const Line3& line) const;

        [[nodiscard]] bool intersectsBox(const Box3& box) const;

        [[nodiscard]] bool intersectsSphere(const Sphere& sphere) const;

        void coplanarPoint(Vector3& target) const;

        Plane& applyMatrix4(const Matrix4& matrix);

        Plane& applyMatrix4(const Matrix4& matrix, Matrix3& normalMatrix);

        Plane& translate(const Vector3& offset);

        [[nodiscard]] bool equals(const Plane& plane) const;

        bool operator==(const Plane& plane) const;
    };

}// namespace threepp

#endif//THREEPP_PLANE_HPP
