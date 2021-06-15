// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix3.js

#ifndef THREEPP_MATRIX3_HPP
#define THREEPP_MATRIX3_HPP

namespace threepp::math {

    class matrix3 {

    public:
        matrix3() = default;

    private:
        const double elements_[9] = {
                1,0,0,
                0,1,0,
                0,0,1
        };

        friend class vector3;
        friend class euler;

    };

}

#endif //THREEPP_MATRIX3_HPP
