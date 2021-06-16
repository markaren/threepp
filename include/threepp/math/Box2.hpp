// https://github.com/mrdoob/three.js/blob/r129/src/math/Box2.js

#ifndef THREEPP_BOX2_HPP
#define THREEPP_BOX2_HPP

#include "Vector2.hpp"

namespace threepp {

    class Box2 {

    public:
        Box2();
        Box2(const Vector2 &min, const Vector2 &max);

        Box2 &set(const Vector2 &min, const Vector2 &max);

        Box2 &makeEmpty();

        [[nodiscard]] bool isEmpty() const;


    private:
        Vector2 min_;
        Vector2 max_;

        static Vector2 _vector;

    };

}

#endif//THREEPP_BOX2_HPP
