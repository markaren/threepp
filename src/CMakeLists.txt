set(publicHeaderDir ${CMAKE_SOURCE_DIR}/include)

set(publicHeaders
        "threepp/math/color.hpp"
        "threepp/math/euler.hpp"
        "threepp/math/matrix3.hpp"
        "threepp/math/matrix4.hpp"
        "threepp/math/plane.hpp"
        "threepp/math/vector2.hpp"
        "threepp/math/vector3.hpp"
        "threepp/math/vector4.hpp"
        "threepp/math/triangle.hpp"
        "threepp/math/quaternion.hpp"
        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
  list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

### MATH

set(sources
        "threepp/math/color.cpp"
        "threepp/math/euler.cpp"
        "threepp/math/matrix3.cpp"
        "threepp/math/matrix4.cpp"
        "threepp/math/plane.cpp"
        "threepp/math/triangle.cpp"
        "threepp/math/vector2.cpp"
        "threepp/math/vector3.cpp"
        "threepp/math/vector4.cpp"
        "threepp/math/quaternion.cpp"
        )

set(privateHeaders
        )

add_library(math ${sources} ${privateHeaders} ${publicHeadersFull})
target_include_directories(math
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        private
        "${CMAKE_CURRENT_SOURCE_DIR}")

