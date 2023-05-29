
#include "threepp/cameras/PerspectiveCamera.hpp"

#include "../equals_util.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace threepp;

TEST_CASE("updateProjectionMatrix") {
    auto cam = PerspectiveCamera(75, 16 / 9.f, 0.1f, 300.0f);

    // updateProjectionMatrix is called in constructor
    auto m = cam.projectionMatrix;

    // perspective projection is given my the 4x4 Matrix
    // 2n/r-l		0			l+r/r-l				 0
    //   0		2n/t-b	t+b/t-b				 0
    //   0			0		-(f+n/f-n)	-(2fn/f-n)
    //   0			0				-1					 0

    // this matrix was calculated by hand via glMatrix.perspective(75, 16 / 9, 0.1, 300.0, pMatrix)
    // to get a reference matrix from plain WebGL
    auto reference = Matrix4().set(
            0.7330642938613892f, 0, 0, 0,
            0, 1.3032253980636597f, 0, 0,
            0, 0, -1.000666856765747f, -0.2000666856765747f,
            0, 0, -1, 0);

    // assert.ok( reference.equals(m) );
    REQUIRE(matrixEquals4(reference, m, 0.000001));
}
