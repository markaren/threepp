// https://github.com/mrdoob/three.js/blob/r129/src/math/Box3.js

#ifndef THREEPP_BOX3_HPP
#define THREEPP_BOX3_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/core/Object3D.hpp"

namespace threepp {

    class Box3 {

    public:
        Box3();
        Box3(Vector3 min, Vector3 max);

        Box3 &set(const Vector3 &min, const Vector3 &max);

        Box3 &makeEmpty();

        [[nodiscard]] bool isEmpty() const;

        void getCenter(Vector3 &target) const;

        void getSize(Vector3 &target) const;

        Box3 &expandByPoint(const Vector3 &point);

        Box3 &expandByVector(const Vector3 &vector);

        Box3 &expandByScalar(float scalar);

        Box3& expandByObject( Object3D &object );


    private:
        Vector3 min_;
        Vector3 max_;
    };

}// namespace threepp

#endif//THREEPP_BOX3_HPP
