
#include "threepp/math/SphericalHarmonics3.hpp"

threepp::SphericalHarmonis3::SphericalHarmonis3() : coefficients_(9) {}

threepp::SphericalHarmonis3 &threepp::SphericalHarmonis3::set(const std::vector<Vector3> &coefficients) {

    for (int i = 0; i < 9; i++) {

        this->coefficients_[i].copy(coefficients[i]);
    }

    return *this;
}

threepp::SphericalHarmonis3 &threepp::SphericalHarmonis3::zero() {

    for (int i = 0; i < 9; i++) {

        this->coefficients_[i].set(0, 0, 0);
    }

    return *this;
}

void threepp::SphericalHarmonis3::getAt(const threepp::Vector3 &normal, threepp::Vector3 &target) {

    // normal is assumed to be unit length

    const auto x = normal.x, y = normal.y, z = normal.z;

    const auto &coeff = this->coefficients_;

    // band 0
    target.copy(coeff[0]).multiply(0.282095f);

    // band 1
    target.addScaledVector(coeff[1], 0.488603f * y);
    target.addScaledVector(coeff[2], 0.488603f * z);
    target.addScaledVector(coeff[3], 0.488603f * x);

    // band 2
    target.addScaledVector(coeff[4], 1.092548f * (x * y));
    target.addScaledVector(coeff[5], 1.092548f * (y * z));
    target.addScaledVector(coeff[6], 0.315392f * (3.0f * z * z - 1.0f));
    target.addScaledVector(coeff[7], 1.092548f * (x * z));
    target.addScaledVector(coeff[8], 0.546274f * (x * x - y * y));
}

void threepp::SphericalHarmonis3::getIrradianceAt(const threepp::Vector3 &normal, threepp::Vector3 &target) {

    // normal is assumed to be unit length

    const auto x = normal.x, y = normal.y, z = normal.z;

    const auto& coeff = this->coefficients_;

    // band 0
    target.copy( coeff[ 0 ] ).multiply( 0.886227f ); // π * 0.282095

    // band 1
    target.addScaledVector( coeff[ 1 ], 2.0f * 0.511664f * y ); // ( 2 * π / 3 ) * 0.488603
    target.addScaledVector( coeff[ 2 ], 2.0f * 0.511664f * z );
    target.addScaledVector( coeff[ 3 ], 2.0f * 0.511664f * x );

    // band 2
    target.addScaledVector( coeff[ 4 ], 2.0f * 0.429043f * x * y ); // ( π / 4 ) * 1.092548
    target.addScaledVector( coeff[ 5 ], 2.0f * 0.429043f * y * z );
    target.addScaledVector( coeff[ 6 ], 0.743125f * z * z - 0.247708f ); // ( π / 4 ) * 0.315392 * 3
    target.addScaledVector( coeff[ 7 ], 2.0f * 0.429043f * x * z );
    target.addScaledVector( coeff[ 8 ], 0.429043f * ( x * x - y * y ) ); // ( π / 4 ) * 0.546274

}

threepp::SphericalHarmonis3 &threepp::SphericalHarmonis3::add(const threepp::SphericalHarmonis3 &sh) {

    for ( int i = 0; i < 9; i ++ ) {

        this->coefficients_[ i ].add( sh.coefficients_[ i ] );

    }

    return *this;

}

threepp::SphericalHarmonis3 &threepp::SphericalHarmonis3::addScaledSH(const threepp::SphericalHarmonis3 &sh, float s) {

    for ( int i = 0; i < 9; i ++ ) {

        this->coefficients_[ i ].addScaledVector( sh.coefficients_[ i ], s );

    }

    return *this;

}

threepp::SphericalHarmonis3 &threepp::SphericalHarmonis3::scale(float s) {

    for ( int i = 0; i < 9; i ++ ) {

        this->coefficients_[ i ].multiply( s );

    }

    return *this;

}

threepp::SphericalHarmonis3 &threepp::SphericalHarmonis3::lerp(const threepp::SphericalHarmonis3 &sh, float alpha) {

    for ( int i = 0; i < 9; i ++ ) {

        this->coefficients_[ i ].lerp( sh.coefficients_[ i ], alpha );

    }

    return *this;

}
