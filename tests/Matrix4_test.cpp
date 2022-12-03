
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/math/Matrix3.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace {

    constexpr float eps = 0.0001;

    bool matrixEquals4(const Matrix4 &a, const Matrix4 &b, float tolerance = eps) {

        for (unsigned i = 0, il = a.elements.size(); i < il; i++) {

            auto delta = a.elements[i] - b.elements[i];
            if (delta > tolerance) {

                return false;
            }
        }

        return true;
    }

    bool eulerEquals( const Euler& a, const Euler& b, float tolerance = eps ) {

        auto diff = std::abs( a.x - b.x ) + std::abs( a.y - b.y ) + std::abs( a.z - b.z );
        return ( diff < tolerance );

    }


}// namespace

TEST_CASE("determinant") {

    Matrix4 a;
    REQUIRE(a.determinant() == Approx(1));

    a.elements[0] = 2;
    REQUIRE(a.determinant() == Approx(2));

    a.elements[0] = 0;
    REQUIRE(a.determinant() == Approx(0));

    // calculated via http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
    a.set(2, 3, 4, 5, -1, -21, -3, -4, 6, 7, 8, 10, -8, -9, -10, -12);
    REQUIRE(a.determinant() == Approx(76));
}

TEST_CASE("set") {
    Matrix4 m;

    m.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    REQUIRE(m.elements[0] == 0);
    REQUIRE(m.elements[1] == 4);
    REQUIRE(m.elements[2] == 8);
    REQUIRE(m.elements[3] == 12);
    REQUIRE(m.elements[4] == 1);
    REQUIRE(m.elements[5] == 5);
    REQUIRE(m.elements[6] == 9);
    REQUIRE(m.elements[7] == 13);
    REQUIRE(m.elements[8] == 2);
    REQUIRE(m.elements[9] == 6);
    REQUIRE(m.elements[10] == 10);
    REQUIRE(m.elements[11] == 14);
    REQUIRE(m.elements[12] == 3);
    REQUIRE(m.elements[13] == 7);
    REQUIRE(m.elements[14] == 11);
    REQUIRE(m.elements[15] == 15);
}

TEST_CASE("identity") {

    Matrix4 a;
    Matrix4 b;

    a.set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    REQUIRE(a.elements[0] == 0);
    REQUIRE(a.elements[1] == 4);
    REQUIRE(a.elements[2] == 8);
    REQUIRE(a.elements[3] == 12);
    REQUIRE(a.elements[4] == 1);
    REQUIRE(a.elements[5] == 5);
    REQUIRE(a.elements[6] == 9);
    REQUIRE(a.elements[7] == 13);
    REQUIRE(a.elements[8] == 2);
    REQUIRE(a.elements[9] == 6);
    REQUIRE(a.elements[10] == 10);
    REQUIRE(a.elements[11] == 14);
    REQUIRE(a.elements[12] == 3);
    REQUIRE(a.elements[13] == 7);
    REQUIRE(a.elements[14] == 11);
    REQUIRE(a.elements[15] == 15);

    REQUIRE(!matrixEquals4( a, b));

    a.identity();

    REQUIRE(matrixEquals4( a, b));
}

TEST_CASE("copy") {
    
    auto a = Matrix4().set( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
    auto b = Matrix4().copy( a );
    
    REQUIRE( matrixEquals4( a, b ));

    // ensure that it is a true copy
    a.elements[ 0 ] = 2;
    REQUIRE( ! matrixEquals4( a, b ));
    
}

TEST_CASE("setFromMatrix3") {

    auto a = Matrix3().set(
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
    );
    auto b = Matrix4();
    auto c = Matrix4().set(
            0, 1, 2, 0,
            3, 4, 5, 0,
            6, 7, 8, 0,
            0, 0, 0, 1
    );
    b.setFromMatrix3( a );
    REQUIRE( b.equals( c ) );

}

TEST_CASE("copyPosition") {

    auto a = Matrix4().set( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
    auto b = Matrix4().set( 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 16 );
    
    REQUIRE( !matrixEquals4( a, b ));

    b.copyPosition( a );
    REQUIRE( matrixEquals4( a, b ));

}

TEST_CASE("makeRotationFromEuler/extractRotation") {

    std::vector<Euler> testValues{
            Euler(0, 0, 0, Euler::RotationOrders::XYZ),
            Euler(1, 0, 0, Euler::RotationOrders::XYZ),
            Euler(0, 1, 0, Euler::RotationOrders::ZYX),
            Euler(0, 0, 0.5, Euler::RotationOrders::YZX),
            Euler(0, 0, -0.5, Euler::RotationOrders::YZX)};

    for (const auto & v : testValues) {

        auto m = Matrix4().makeRotationFromEuler( v );

        auto v2 = Euler().setFromRotationMatrix( m, v.getOrder() );
        auto m2 = Matrix4().makeRotationFromEuler( v2 );

        REQUIRE( matrixEquals4( m, m2, eps ));
        REQUIRE( eulerEquals( v, v2, eps ));

        auto m3 = Matrix4().extractRotation( m2 );
        auto v3 = Euler().setFromRotationMatrix( m3, v.getOrder() );

        REQUIRE( matrixEquals4( m, m3, eps ));
        REQUIRE( eulerEquals( v, v3, eps ));

    }

}

TEST_CASE("lookat") {

    Matrix4 a;
    Matrix4 expected = Matrix4().identity();
    Vector3 eye(0, 0, 0);
    Vector3 target(0, 1, -1);
    Vector3 up(0, 1, 0);

    a.lookAt(eye, target, up);
    auto rotation = Euler().setFromRotationMatrix(a);
    REQUIRE(rotation.x * (180 / math::PI) == Approx(45));

    // eye and target are in the same position
    eye.copy(target);
    a.lookAt(eye, target, up);
    REQUIRE(a == expected);

    // up and z are parallel
    eye.set(0, 1, 0);
    target.set(0, 0, 0);
    a.lookAt(eye, target, up);
    expected.set(
            1, 0, 0, 0,
            0, 0.0001, 1, 0,
            0, -1, 0.0001, 0,
            0, 0, 0, 1);
    REQUIRE(a == expected);
}

TEST_CASE("premultiply") {

    auto lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53);
    auto rhs = Matrix4().set(59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131);

    rhs.premultiply(lhs);

    REQUIRE(rhs.elements[0] == Approx(1585));
    REQUIRE(rhs.elements[1] == Approx(5318));
    REQUIRE(rhs.elements[2] == Approx(10514));
    REQUIRE(rhs.elements[3] == Approx(15894));
    REQUIRE(rhs.elements[4] == Approx(1655));
    REQUIRE(rhs.elements[5] == Approx(5562));
    REQUIRE(rhs.elements[6] == Approx(11006));
    REQUIRE(rhs.elements[7] == Approx(16634));
    REQUIRE(rhs.elements[8] == Approx(1787));
    REQUIRE(rhs.elements[9] == Approx(5980));
    REQUIRE(rhs.elements[10] == Approx(11840));
    REQUIRE(rhs.elements[11] == Approx(17888));
    REQUIRE(rhs.elements[12] == Approx(1861));
    REQUIRE(rhs.elements[13] == Approx(6246));
    REQUIRE(rhs.elements[14] == Approx(12378));
    REQUIRE(rhs.elements[15] == Approx(18710));
}

TEST_CASE("transpose") {

    Matrix4 a;
    Matrix4 b = Matrix4(a).transpose();
    REQUIRE(matrixEquals4( a, b));

    b = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    Matrix4 c = Matrix4(b).transpose();
    REQUIRE(!matrixEquals4( b, c));
    c.transpose();
    REQUIRE(matrixEquals4( b, c));
}

TEST_CASE("multipyMatrices") {

    auto lhs = Matrix4().set(2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53);
    auto rhs = Matrix4().set(59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131);
    auto ans = Matrix4();

    ans.multiplyMatrices(lhs, rhs);

    REQUIRE(ans.elements[0] == Approx(1585));
    REQUIRE(ans.elements[1] == Approx(5318));
    REQUIRE(ans.elements[2] == Approx(10514));
    REQUIRE(ans.elements[3] == Approx(15894));
    REQUIRE(ans.elements[4] == Approx(1655));
    REQUIRE(ans.elements[5] == Approx(5562));
    REQUIRE(ans.elements[6] == Approx(11006));
    REQUIRE(ans.elements[7] == Approx(16634));
    REQUIRE(ans.elements[8] == Approx(1787));
    REQUIRE(ans.elements[9] == Approx(5980));
    REQUIRE(ans.elements[10] == Approx(11840));
    REQUIRE(ans.elements[11] == Approx(17888));
    REQUIRE(ans.elements[12] == Approx(1861));
    REQUIRE(ans.elements[13] == Approx(6246));
    REQUIRE(ans.elements[14] == Approx(12378));
    REQUIRE(ans.elements[15] == Approx(18710));
}

TEST_CASE("invert") {

    auto zero = Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    auto identity = Matrix4();

    auto a = Matrix4();
    auto b = Matrix4().set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    a.copy(b).invert();
    REQUIRE(matrixEquals4( a, zero));

    std::vector<Matrix4> testMatrices = {
            Matrix4().makeRotationX(0.3),
            Matrix4().makeRotationX(-0.3),
            Matrix4().makeRotationY(0.3),
            Matrix4().makeRotationY(-0.3),
            Matrix4().makeRotationZ(0.3),
            Matrix4().makeRotationZ(-0.3),
            Matrix4().makeScale(1, 2, 3),
            Matrix4().makeScale(1 / 8., 1 / 2., 1 / 3.),
            Matrix4().makePerspective(-1, 1, 1, -1, 1, 1000),
            Matrix4().makePerspective(-16, 16, 9, -9, 0.1, 10000),
            Matrix4().makeTranslation(1, 2, 3)};

    for (const auto &m : testMatrices) {

        auto mInverse = Matrix4().copy(m).invert();
        auto mSelfInverse = Matrix4(m);
        mSelfInverse.copy(mSelfInverse).invert();

        // self-inverse should the same as inverse
        REQUIRE(matrixEquals4( mSelfInverse, mInverse));

        // the determinant of the inverse should be the reciprocal
        REQUIRE(std::abs((m.determinant() * mInverse.determinant()) - 1) < 0.0001);

        auto mProduct = Matrix4().multiplyMatrices(m, mInverse);

        // the determinant of the identity matrix is 1
        REQUIRE(std::abs(mProduct.determinant() - 1) < 0.0001);
        REQUIRE(matrixEquals4(mProduct, identity));
    }
}

TEST_CASE("scale") {

    auto a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto b = Vector3(2, 3, 4);
    auto c = Matrix4().set(2, 6, 12, 4, 10, 18, 28, 8, 18, 30, 44, 12, 26, 42, 60, 16);

    a.scale(b);
    REQUIRE(matrixEquals4( a, c));
}

TEST_CASE("getMaxScaleOnAxis") {

    auto a = Matrix4().set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    auto expected = std::sqrt(3 * 3 + 7 * 7 + 11 * 11);

    REQUIRE(std::abs(a.getMaxScaleOnAxis() - expected) <= eps);
}

TEST_CASE("makeScale") {

    auto a = Matrix4();
    auto c = Matrix4().set(2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1);

    a.makeScale(2, 3, 4);
    REQUIRE(matrixEquals4( a, c));
}

TEST_CASE("makeShear") {

    auto a = Matrix4();
    auto c = Matrix4().set(1, 3, 5, 0, 1, 1, 6, 0, 2, 4, 1, 0, 0, 0, 0, 1);

    a.makeShear(1, 2, 3, 4, 5, 6);
    REQUIRE(matrixEquals4( a, c));
}

TEST_CASE("makePerspective") {

    auto a = Matrix4().makePerspective(-1, 1, -1, 1, 1, 100);
    auto expected = Matrix4().set(
            1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, -101 / 99., -200 / 99.,
            0, 0, -1, 0);
    REQUIRE(matrixEquals4( a, expected));
}

TEST_CASE("equals") {

    auto a = Matrix4().set(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    auto b = Matrix4().set(0, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    REQUIRE(!a.equals(b));
    REQUIRE(!b.equals(a));

    a.copy(b);
    REQUIRE(a.equals(b));
    REQUIRE(b.equals(a));
}
