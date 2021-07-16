
#ifndef THREEPP_FACE3_HPP
#define THREEPP_FACE3_HPP

#include "threepp/math/Vector3.hpp"

namespace threepp {

    class Face3 {

    public:
        int a;
        int b;
        int c;
        Vector3 normal;

        int materialIndex;

        Face3(int a, int b, int c, const Vector3 &normal, int materialIndex)
            : a(a), b(b), c(c), normal(normal), materialIndex(materialIndex) {}
    };

}

#endif//THREEPP_FACE3_HPP
