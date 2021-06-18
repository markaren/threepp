// https://github.com/mrdoob/three.js/blob/r129/src/math/SphericalHarmonics3.js

#ifndef THREEPP_SPHERICALHARMONICS3_HPP
#define THREEPP_SPHERICALHARMONICS3_HPP

#include "threepp/math/Vector3.hpp"

#include <vector>

namespace threepp {

    class SphericalHarmonis3 {

    public:
        SphericalHarmonis3();

        SphericalHarmonis3 &set(const std::vector<Vector3> &coefficients);

        SphericalHarmonis3 &zero();

        // get the radiance in the direction of the normal
        // target is a Vector3
        void getAt(const Vector3 &normal, Vector3 &target);

    private:
        std::vector<Vector3> coefficients_;
    };

}// namespace threepp

#endif//THREEPP_SPHERICALHARMONICS3_HPP
