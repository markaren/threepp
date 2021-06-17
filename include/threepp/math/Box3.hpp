// https://github.com/mrdoob/three.js/blob/r129/src/math/Box3.js

#ifndef THREEPP_BOX3_HPP
#define THREEPP_BOX3_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/math/Triangle.hpp"
#include "threepp/math/Vector3.hpp"

#include <array>

namespace threepp {

    class Box3 {

    public:
        Box3();
        Box3(Vector3 min, Vector3 max);

        Box3 &set(const Vector3 &min, const Vector3 &max);

        Box3 &setFromPoints(const std::vector<Vector3> &points);

        Box3 &setFromCenterAndSize(const Vector3 &center, const Vector3 &size);

        Box3 &copy(const Box3 &box);

        Box3 &makeEmpty();

        [[nodiscard]] bool isEmpty() const;

        void getCenter(Vector3 &target) const;

        void getSize(Vector3 &target) const;

        Box3 &expandByPoint(const Vector3 &point);

        Box3 &expandByVector(const Vector3 &vector);

        Box3 &expandByScalar(float scalar);

        [[nodiscard]] bool containsPoint(const Vector3 &point) const;

        [[nodiscard]] bool containsBox(const Box3 &box) const;

        void getParameter(const Vector3 &point, Vector3 &target) const;

        [[nodiscard]] bool intersectsBox(const Box3 &box) const;

        bool intersectsSphere(const Sphere &sphere);

        bool intersectsPlane(const Plane &plane);

        bool intersectsTriangle(const Triangle &triangle);

        Vector3 &clampPoint(const Vector3 &point, Vector3 &target) const;

        [[nodiscard]] float distanceToPoint(const Vector3 &point) const;

    private:
        Vector3 min_;
        Vector3 max_;

        static std::array<Vector3, 9> _points;

        static Vector3 _vector;

        static Box3 _box;

        static Vector3 _v0;
        static Vector3 _v1;
        static Vector3 _v2;

        static Vector3 _f0;
        static Vector3 _f1;
        static Vector3 _f2;

        static Vector3 _center;
        static Vector3 _extents;
        static Vector3 _triangleNormal;
        static Vector3 _testAxis;

        static bool satForAxes(const std::vector<float> &axes, const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, const Vector3 &extents);
    };

}// namespace threepp

#endif//THREEPP_BOX3_HPP
