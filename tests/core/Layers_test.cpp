
#include <catch2/catch_test_macros.hpp>

#include "threepp/core/Layers.hpp"

using namespace threepp;

TEST_CASE("Test enable") {

    Layers a;

    a.set(0);
    a.enable(0);
    CHECK(a.mask() == 1);

    a.set(0);
    a.enable(1);
    CHECK(a.mask() == 3);

    a.set(1);
    a.enable(0);
    CHECK(a.mask() == 3);

    a.set(1);
    a.enable(1);
    CHECK(a.mask() == 2);
}

TEST_CASE("Test toggle") {

    Layers a;

    a.set(0);
    a.toggle(0);
    CHECK(a.mask() == 0);

    a.set(0);
    a.toggle(1);
    CHECK(a.mask() == 3);

    a.set(1);
    a.toggle(0);
    CHECK(a.mask() == 3);

    a.set(1);
    a.toggle(1);
    CHECK(a.mask() == 0);
}

TEST_CASE("Test disable") {

    Layers a;

    a.set(0);
    a.disable(0);
    CHECK(a.mask() == 0);

    a.set(0);
    a.disable(1);
    CHECK(a.mask() == 1);

    a.set(1);
    a.disable(0);
    CHECK(a.mask() == 2);

    a.set(1);
    a.disable(1);
    CHECK(a.mask() == 0);
}

TEST_CASE("Test test") {

    Layers a;
    Layers b;

    CHECK(a.test(b));

    a.set(1);
    CHECK(!a.test(b));

    b.toggle(1);
    CHECK(a.test(b));
}

TEST_CASE("Test isEnabled") {

    Layers a;

    a.enable(1);
    CHECK(a.isEnabled(1));

    a.enable(2);
    CHECK(a.isEnabled(2));

    a.toggle(1);
    CHECK(!a.isEnabled(1));
    CHECK(a.isEnabled(2));
}
