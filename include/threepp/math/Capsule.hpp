// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/math/Capsule.js

#ifndef THREEPP_CAPSULE_HPP
#define THREEPP_CAPSULE_HPP

#include "threepp/math/Vector3.hpp"

#include <utility>

namespace threepp {

    class Box3;
    class Line3;

    class Capsule {

        Vector3 start;
        Vector3 end;
        float radius;

        explicit Capsule(const Vector3& start = {0, 0, 0}, const Vector3& end = {0, 1, 0}, float radius = 1);

        [[nodiscard]] Capsule clone() const;

        void set(const Vector3& start, const Vector3& end, float radius);

        void copy(const Capsule& capsule);

        void getCenter(Vector3& target) const;

        void translate(const Vector3& v);

        static bool checkAABBAxis(float p1x, float p1y, float p2x, float p2y, float minx, float maxx, float miny, float maxy, float radius);

        [[nodiscard]] bool intersectsBox(const Box3& box) const;

        [[nodiscard]] std::pair<Vector3, Vector3> lineLineMinimumPoints(const Line3& line1, const Line3& line2) const;
    };

}// namespace threepp

#endif//THREEPP_CAPSULE_HPP
