// https://github.com/mrdoob/three.js/blob/r129/src/math/Frustum.js

#ifndef THREEPP_FRUSTUM_HPP
#define THREEPP_FRUSTUM_HPP

#include "threepp/math/Plane.hpp"

#include <array>

namespace threepp {

    class Sphere;

    class Frustum {

    public:
        explicit Frustum(Plane p0 = Plane(), Plane p1 = Plane(), Plane p2 = Plane(), Plane p3 = Plane(), Plane p4 = Plane(), Plane p5 = Plane());

        Frustum &set(const Plane &p0, const Plane &p1, const Plane &p2, const Plane &p3, const Plane &p4, const Plane &p5) {

            planes_[0].copy(p0);
            planes_[1].copy(p1);
            planes_[2].copy(p2);
            planes_[3].copy(p3);
            planes_[4].copy(p4);
            planes_[5].copy(p5);

            return *this;
        }

        Frustum &copy( const Frustum &frustum );

        Frustum &setFromProjectionMatrix( const Matrix4 &m );



    private:
        std::array<Plane, 6> planes_;

        static Sphere _sphere;
        static Vector3 _vector;
    };

}// namespace threepp

#endif//THREEPP_FRUSTUM_HPP
