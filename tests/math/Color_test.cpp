
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/math/Color.hpp"

using namespace threepp;


TEST_CASE("instancing") {

    // rgb ctor
    Color c(1, 1, 1);
    CHECK(c.r == 1);
    CHECK(c.g == 1);
    CHECK(c.b == 1);
}

TEST_CASE("set") {

    Color a;
    Color b( 0.5, 0, 0 );
    Color c( 0xFF0000 );
    Color d( 0, 1.0, 0 );

    a.copy( b );
    CHECK( a.equals( b ));

    a.setHex( 0xFF0000 );
    CHECK(a.equals( c ));

}

TEST_CASE("setHSL") {

    Color c;
    HSL hsl = { 0, 0, 0 };
    c.setHSL( 0.75, 1.0, 0.25 );
    c.getHSL( hsl );

    assert.ok( hsl.h == 0.75, "hue: " + hsl.h );
    assert.ok( hsl.s == 1.00, "saturation: " + hsl.s );
    assert.ok( hsl.l == 0.25, "lightness: " + hsl.l );

}
