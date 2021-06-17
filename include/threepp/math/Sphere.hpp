// https://github.com/mrdoob/three.js/blob/r129/src/math/Sphere.js

#ifndef THREEPP_SPHERE_HPP
#define THREEPP_SPHERE_HPP

#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace threepp {

    class Box3;

    class Sphere {

    public:

        Vector3 center;
        float radius;

        explicit Sphere(Vector3 center = Vector3(), float radius = -1);

        Sphere &set(const Vector3 &center, float radius);

        Sphere &setFromPoints(const std::vector<Vector3> &points, Vector3 *optionalCenter = nullptr);

    private:


        static Box3 _box;
        static Vector3 _v1;
    };

}// namespace threepp

#endif//THREEPP_SPHERE_HPP
