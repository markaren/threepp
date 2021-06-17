// https://github.com/mrdoob/three.js/blob/r129/src/math/Sphere.js

#ifndef THREEPP_SPHERE_HPP
#define THREEPP_SPHERE_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace threepp {

    class Sphere {

    public:
        explicit Sphere(Vector3 center, float radius = -1) : center_(center), radius_(radius) {}

        Sphere &set(const Vector3 &center, float radius) {

            this->center_ = (center);
            this->radius_ = radius;

            return *this;
        }

        Sphere &setFromPoints(const std::vector<Vector3> &points, Vector3 *optionalCenter = nullptr) {

            if (optionalCenter) {

                center_ = (*optionalCenter);

            } else {

                _box.setFromPoints(points).getCenter(center_);
            }

            float maxRadiusSq = 0;

            for (auto &point : points) {

                maxRadiusSq = std::max(maxRadiusSq, center_.distanceToSquared(point));
            }

            this->radius_ = std::sqrt(maxRadiusSq);

            return *this;
        }


    private:
        Vector3 center_;
        float radius_;

        inline static Box3 _box = Box3();
        inline static Vector3 _v1 = Vector3();
    };

}// namespace threepp

#endif//THREEPP_SPHERE_HPP
