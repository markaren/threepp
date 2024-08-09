
#include "threepp/cameras/OrthographicCamera.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace threepp;

TEST_CASE("updateProjectionMatrix") {
    float left = -1, right = 1, top = 1, bottom = -1;
    float near = 1, far = 3;
    auto cam = OrthographicCamera::create(left, right, top, bottom, near, far);

    // updateProjectionMatrix is called in constructor
    const auto& pMatrix = cam->projectionMatrix.elements;

    // orthographic projection is given my the 4x4 Matrix
    // 2/r-l		0			 0		-(l+r/r-l)
    //   0		2/t-b		 0		-(t+b/t-b)
    //   0			0		-2/f-n	-(f+n/f-n)
    //   0			0			 0				1

    CHECK_THAT(pMatrix[0], Catch::Matchers::WithinRel (2.f / (right - left)));
    CHECK_THAT(pMatrix[5], Catch::Matchers::WithinRel (2.f / (top - bottom)));
    CHECK_THAT(pMatrix[10], Catch::Matchers::WithinRel (-2 / (far - near)));
    CHECK_THAT(pMatrix[12], Catch::Matchers::WithinRel (-((right + left) / (right - left))));
    CHECK_THAT(pMatrix[13], Catch::Matchers::WithinRel ( -((top + bottom) / (top - bottom))));
    CHECK_THAT(pMatrix[14], Catch::Matchers::WithinRel (-((far + near) / (far - near))));
}
