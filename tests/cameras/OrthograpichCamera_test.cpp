
#include "threepp/cameras/OrthographicCamera.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace threepp;

TEST_CASE("updateProjectionMatrix") {
    int left = -1, right = 1, top = 1, bottom = -1;
    float near = 1, far = 3;
    auto cam = OrthographicCamera::create(left, right, top, bottom, near, far);

    // updateProjectionMatrix is called in constructor
    const auto& pMatrix = cam->projectionMatrix.elements;

    // orthographic projection is given my the 4x4 Matrix
    // 2/r-l		0			 0		-(l+r/r-l)
    //   0		2/t-b		 0		-(t+b/t-b)
    //   0			0		-2/f-n	-(f+n/f-n)
    //   0			0			 0				1

    CHECK(pMatrix[0] == 2.f / (right - left));
    CHECK(pMatrix[5] == 2.f / (top - bottom));
    CHECK(pMatrix[10] == -2 / (far - near));
    CHECK(pMatrix[12] == -((right + left) / (right - left)));
    CHECK(pMatrix[13] == -((top + bottom) / (top - bottom)));
    CHECK(pMatrix[14] == -((far + near) / (far - near)));
}
