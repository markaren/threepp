// https://github.com/mrdoob/three.js/blob/r129/src/math/Ray.js

#ifndef THREEPP_RAY_HPP
#define THREEPP_RAY_HPP

#include "threepp/math/Vector3.hpp"

#include <optional>

namespace threepp {

    class Sphere;
    class Plane;
    class Box3;

    class Ray {

    public:
        Vector3 origin;
        Vector3 direction;

        explicit Ray(const Vector3& origin = Vector3(), const Vector3& direction = Vector3(0, 0, -1));

        Ray& set(const Vector3& origin, const Vector3& direction);

        Ray& copy(const Ray& ray);

        Vector3& at(float t, Vector3& target) const;

        Ray& lookAt(const Vector3& v);

        Ray& recast(float t);

        void closestPointToPoint(const Vector3& point, Vector3& target) const;

        [[nodiscard]] float distanceToPoint(const Vector3& point) const;

        [[nodiscard]] float distanceSqToPoint(const Vector3& point) const;

        [[nodiscard]] float distanceSqToSegment(const Vector3& v0, const Vector3& v1, Vector3* optionalPointOnRay = nullptr, Vector3* optionalPointOnSegment = nullptr) const;

        void intersectSphere(const Sphere& sphere, Vector3& target) const;

        [[nodiscard]] bool intersectsSphere(const Sphere& sphere) const;

        [[nodiscard]] float distanceToPlane(const Plane& plane) const;

        void intersectPlane(const Plane& plane, Vector3& target) const;

        [[nodiscard]] bool intersectsPlane(const Plane& plane) const;

        void intersectBox(const Box3& box, Vector3& target) const;

        [[nodiscard]] bool intersectsBox(const Box3& box) const;

        std::optional<Vector3> intersectTriangle(const Vector3& a, const Vector3& b, const Vector3& c, bool backfaceCulling, Vector3& target) const;

        Ray& applyMatrix4(const Matrix4& matrix4);
    };

}// namespace threepp

#endif//THREEPP_RAY_HPP
