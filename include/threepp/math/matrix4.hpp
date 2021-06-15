// https://github.com/mrdoob/three.js/blob/r129/src/math/Matrix4.js

#ifndef THREEPP_MATRIX4_HPP
#define THREEPP_MATRIX4_HPP

namespace threepp::math {

    class matrix4 {

    public:
        matrix4() = default;


        matrix4 &set( double n11, double n12, double n13, double n14, double n21, double n22, double n23, double n24, double n31, double n32, double n33, double n34, double n41, double n42, double n43, double n44 ) {

          auto te = this->elements_; //TODO

            te[ 0 ] = n11; te[ 4 ] = n12; te[ 8 ] = n13; te[ 12 ] = n14;
            te[ 1 ] = n21; te[ 5 ] = n22; te[ 9 ] = n23; te[ 13 ] = n24;
            te[ 2 ] = n31; te[ 6 ] = n32; te[ 10 ] = n33; te[ 14 ] = n34;
            te[ 3 ] = n41; te[ 7 ] = n42; te[ 11 ] = n43; te[ 15 ] = n44;

            return *this;

        }

        matrix4 &identity() {

            this->set(

                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1

            );

            return *this;

        }


    private:
        double elements_[16] = {
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1
        };

        friend class vector3;

    };

}

#endif //THREEPP_MATRIX4_HPP
