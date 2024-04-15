
#include <catch2/catch_test_macros.hpp>

#include "threepp/math/Box2.hpp"
#include "threepp/math/infinity.hpp"

using namespace threepp;

namespace {

    const Vector2 zero2(0, 0);
    const Vector2 one2(1, 1);
    const Vector2 two2(2, 2);
    const Vector2 negOne2(-1, -1);
    const Vector2 posInf2(+Infinity<float>, +Infinity<float>);
    const Vector2 negInf2(-Infinity<float>, -Infinity<float>);

}// namespace

TEST_CASE("instancing") {

    Box2 a;
    CHECK(a.getMin().equals(posInf2));
    CHECK(a.getMax().equals(negInf2));

    a = Box2(zero2, zero2);
    CHECK(a.getMin().equals(zero2));
    CHECK(a.getMax().equals(zero2));

    a = Box2(zero2, one2);
    CHECK(a.getMin().equals(zero2));
    CHECK(a.getMax().equals(one2));
}

TEST_CASE("set") {

    Box2 a;

    a.set(zero2, one2);
    CHECK(a.getMin().equals(zero2));
    CHECK(a.getMax().equals(one2));
}

TEST_CASE("setFromPoints") {

    Box2 a;

    a.setFromPoints({zero2, one2, two2});
    CHECK(a.getMin().equals(zero2));
    CHECK(a.getMax().equals(two2));

    a.setFromPoints({one2});
    CHECK(a.getMin().equals(one2));
    CHECK(a.getMax().equals(one2));

    a.setFromPoints({});
    CHECK(a.isEmpty());
}

TEST_CASE("setFromCenterAndSize") {

    Box2 a;

    a.setFromCenterAndSize(zero2, two2);
    CHECK(a.getMin().equals(negOne2));
    CHECK(a.getMax().equals(one2));

    a.setFromCenterAndSize(one2, two2);
    CHECK(a.getMin().equals(zero2));
    CHECK(a.getMax().equals(two2));

    a.setFromCenterAndSize(zero2, zero2);
    CHECK(a.getMin().equals(zero2));
    CHECK(a.getMax().equals(zero2));
}

TEST_CASE("empty/makeEmpty") {

    Box2 a;

    CHECK(a.isEmpty());

    a = Box2(zero2, one2);
    CHECK(!a.isEmpty());

    a.makeEmpty();
    CHECK(a.isEmpty());
}

TEST_CASE("isEmpty") {

    Box2 a(zero2, zero2);
    CHECK(!a.isEmpty());

    a = Box2(zero2, one2);
    CHECK(!a.isEmpty());

    a = Box2(two2, one2);
    CHECK(a.isEmpty());

    a = Box2(posInf2, negInf2);
    CHECK(a.isEmpty());
}

TEST_CASE("getCenter") {

    Box2 a(zero2, zero2);
    Vector2 center;
    a.getCenter(center);
    CHECK(center.equals(zero2));

    a = Box2(zero2, one2);
    Vector2 midpoint = one2 * 0.5f;
    a.getCenter(center);
    CHECK(center.equals(midpoint));
}

TEST_CASE("getSize") {

    Box2 a = Box2(zero2, zero2);
    Vector2 size;
    a.getSize(size);
    CHECK(size.equals(zero2));

    a = Box2(zero2, one2);
    a.getSize(size);
    CHECK(size.equals(one2));
}

TEST_CASE("expandByPoint") {

    Box2 a(zero2, zero2);
    Vector2 size;
    Vector2 center;

    a.expandByPoint(zero2);
    a.getSize(size);
    CHECK(size.equals(zero2));

    a.expandByPoint(one2);
    a.getSize(size);
    CHECK(size.equals(one2));

    a.expandByPoint(one2.clone().negate());
    a.getSize(size);
    CHECK(size.equals(one2 * 2));
    a.getCenter(center);
    CHECK(center.equals(zero2));
}

TEST_CASE("containsPoint") {
    Box2 a(zero2, zero2);

    CHECK(a.containsPoint(zero2));
    CHECK(!a.containsPoint(one2));

    a.expandByScalar(1);
    CHECK(a.containsPoint(zero2));
    CHECK(a.containsPoint(one2));
    CHECK(a.containsPoint(one2.clone().negate()));
}

TEST_CASE("intersectsBox") {
    Box2 a(zero2, zero2);
    Box2 b(zero2, one2);
    Box2 c(one2.clone().negate(), one2);

    CHECK(a.intersectsBox(a));
    CHECK(a.intersectsBox(b));
    CHECK(a.intersectsBox(c));

    CHECK(b.intersectsBox(a));
    CHECK(c.intersectsBox(a));
    CHECK(b.intersectsBox(c));

    b.translate(two2);
    CHECK(!a.intersectsBox(b));
    CHECK(!b.intersectsBox(a));
    CHECK(!b.intersectsBox(c));
}

TEST_CASE("intersect") {

    Box2 a(zero2, zero2);
    Box2 b(zero2, one2);
    Box2 c(one2.clone().negate(), one2);

    CHECK(a.clone().intersect(a).equals(a));
    CHECK(a.clone().intersect(b).equals(a));
    CHECK(b.clone().intersect(b).equals(b));
    CHECK(a.clone().intersect(c).equals(a));
    CHECK(b.clone().intersect(c).equals(b));
    CHECK(c.clone().intersect(c).equals(c));

    Box2 d(one2.clone().negate(), zero2);
    Box2 e(one2, two2);
    e.intersect(d);

    CHECK((e.getMin().equals(posInf2) && e.getMax().equals(negInf2)));
}
