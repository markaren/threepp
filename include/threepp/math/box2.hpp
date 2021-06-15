// https://github.com/mrdoob/three.js/blob/r129/src/math/Box2.js

#ifndef THREEPP_BOX2_HPP
#define THREEPP_BOX2_HPP

#include "vector2.hpp"

namespace threepp {

    class box2 {

    public:

        box2();
        box2(const vector2 &min, const vector2 &max);

        box2 &set(const vector2 &min, const vector2 &max);

        box2 &makeEmpty();

        [[nodiscard]] bool isEmpty() const;


    private:
        vector2 min_;
        vector2 max_;

        static vector2 _vector;

    };

}

#endif//THREEPP_BOX2_HPP
