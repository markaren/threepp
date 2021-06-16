set(publicHeaderDir ${CMAKE_SOURCE_DIR}/include)

set(publicHeaders
        "threepp/core/BufferAttribute.hpp"
        "threepp/core/BufferGeometry.hpp"
        "threepp/core/Clock.hpp"
        "threepp/core/Object3d.hpp"

        "threepp/math/Box2.hpp"
        "threepp/math/Box3.hpp"
        "threepp/math/Color.hpp"
        "threepp/math/Euler.hpp"
        "threepp/math/Matrix3.hpp"
        "threepp/math/Matrix4.hpp"
        "threepp/math/Plane.hpp"
        "threepp/math/Vector2.hpp"
        "threepp/math/Vector3.hpp"
        "threepp/math/Vector4.hpp"
        "threepp/math/Triangle.hpp"
        "threepp/math/Quaternion.hpp"
        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
  list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

### CORE


### MATH

set(sources
        "threepp/math/Box2.cpp"
        "threepp/math/Box3.cpp"
        "threepp/math/Color.cpp"
        "threepp/math/Euler.cpp"
        "threepp/math/Matrix3.cpp"
        "threepp/math/Matrix4.cpp"
        "threepp/math/Plane.cpp"
        "threepp/math/Triangle.cpp"
        "threepp/math/Vector2.cpp"
        "threepp/math/Vector3.cpp"
        "threepp/math/Vector4.cpp"
        "threepp/math/Quaternion.cpp"
        )

set(privateHeaders
        )

add_library(math ${sources} ${privateHeaders} ${publicHeadersFull})
target_include_directories(math
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        private
        "${CMAKE_CURRENT_SOURCE_DIR}")

