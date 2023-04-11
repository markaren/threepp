// https://github.com/mrdoob/three.js/blob/r129/src/math/Sphere.js

#ifndef THREEPP_SPHERE_HPP
#define THREEPP_SPHERE_HPP

#include "threepp/math/Vector3.hpp"

#include <vector>

namespace threepp {

    class Box3;
    class Plane;

    class Sphere {

    public:
        Vector3 center;
        float radius;

        explicit Sphere(const Vector3& center = Vector3(), float radius = -1);

        Sphere& set(const Vector3& center, float radius);

        Sphere& setFromPoints(const std::vector<Vector3>& points, Vector3* optionalCenter = nullptr);

        Sphere& copy(const Sphere& sphere);

        [[nodiscard]] bool isEmpty() const;

        Sphere& makeEmpty();

        [[nodiscard]] bool containsPoint(const Vector3& point) const;

        [[nodiscard]] float distanceToPoint(const Vector3& point) const;

        [[nodiscard]] bool intersectsSphere(const Sphere& sphere) const;

        [[nodiscard]] bool intersectsBox(const Box3& box) const;

        [[nodiscard]] bool intersectsPlane(const Plane& plane) const;

        void clampPoint(const Vector3& point, Vector3& target) const;

        void getBoundingBox(Box3& target) const;

        Sphere& applyMatrix4(const Matrix4& matrix);

        Sphere& translate(const Vector3& offset);

        Sphere& expandByPoint(const Vector3& point);

        Sphere& union_(const Sphere& sphere);

        [[nodiscard]] bool equals(const Sphere& sphere) const;

        [[nodiscard]] Sphere clone() const;
    };

}// namespace threepp

#endif//THREEPP_SPHERE_HPP
