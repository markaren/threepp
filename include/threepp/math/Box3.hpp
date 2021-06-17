// https://github.com/mrdoob/three.js/blob/r129/src/math/Box3.js

#ifndef THREEPP_BOX3_HPP
#define THREEPP_BOX3_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"

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

        [[nodiscard]] bool containsPoint( const Vector3 &point ) const;

        [[nodiscard]] bool containsBox( const Box3 &box ) const;

        void getParameter( const Vector3 &point, Vector3 &target ) const;

        [[nodiscard]] bool intersectsBox( const Box3 &box ) const;


    private:
        Vector3 min_;
        Vector3 max_;

        static Vector3 _vector;
    };

}// namespace threepp

#endif//THREEPP_BOX3_HPP
