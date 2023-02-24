// https://github.com/mrdoob/three.js/blob/r129/src/math/SphericalHarmonics3.js

#ifndef THREEPP_SPHERICALHARMONICS3_HPP
#define THREEPP_SPHERICALHARMONICS3_HPP

#include "threepp/math/Vector3.hpp"

#include <vector>

namespace threepp {

    class SphericalHarmonis3 {

    public:
        SphericalHarmonis3();

        SphericalHarmonis3& set(const std::vector<Vector3>& coefficients);

        SphericalHarmonis3& zero();

        // get the radiance in the direction of the normal
        // target is a Vector3
        void getAt(const Vector3& normal, Vector3& target);

        // get the irradiance (radiance convolved with cosine lobe) in the direction of the normal
        // target is a Vector3
        // https://graphics.stanford.edu/papers/envmap/envmap.pdf
        void getIrradianceAt(const Vector3& normal, Vector3& target);

        SphericalHarmonis3& add(const SphericalHarmonis3& sh);

        SphericalHarmonis3& addScaledSH(const SphericalHarmonis3& sh, float s);

        SphericalHarmonis3& scale(float s);

        SphericalHarmonis3& lerp(const SphericalHarmonis3& sh, float alpha);

        const std::vector<Vector3>& getCoefficients() const;

    private:
        std::vector<Vector3> coefficients_;
    };

}// namespace threepp

#endif//THREEPP_SPHERICALHARMONICS3_HPP
