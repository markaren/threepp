set(publicHeaderDir ${CMAKE_SOURCE_DIR}/include)

set(publicHeaders

        "threepp/Canvas.hpp"

        "threepp/core/BufferAttribute.hpp"
        "threepp/core/BufferGeometry.hpp"
        "threepp/core/Clock.hpp"
        "threepp/core/Object3D.hpp"

        "threepp/geometries/BoxGeometry.hpp"
        "threepp/geometries/ConeGeometry.hpp"
        "threepp/geometries/CylinderGeometry.hpp"
        "threepp/geometries/PlaneGeometry.hpp"
        "threepp/geometries/SphereGeometry.hpp"

        "threepp/math/Box2.hpp"
        "threepp/math/Box3.hpp"
        "threepp/math/Color.hpp"
        "threepp/math/Cylindrical.hpp"
        "threepp/math/Euler.hpp"
        "threepp/math/Frustum.hpp"
        "threepp/math/Line3.hpp"
        "threepp/math/MathUtils.hpp"
        "threepp/math/Matrix3.hpp"
        "threepp/math/Matrix4.hpp"
        "threepp/math/Plane.hpp"
        "threepp/math/Ray.hpp"
        "threepp/math/Sphere.hpp"
        "threepp/math/Spherical.hpp"
        "threepp/math/SphericalHarmonics3.hpp"
        "threepp/math/Triangle.hpp"
        "threepp/math/Vector2.hpp"
        "threepp/math/Vector3.hpp"
        "threepp/math/Vector4.hpp"
        "threepp/math/Quaternion.hpp"

        "threepp/objects/Line.hpp"
        "threepp/objects/Mesh.hpp"

        "threepp/textures/Image.hpp"
        "threepp/textures/Texture.hpp"

        "threepp/lights/Light.hpp"
        "threepp/lights/LightProbe.hpp"
        "threepp/lights/AmbientLight.hpp"
        "threepp/lights/DirectionalLight.hpp"
        "threepp/lights/LightShadow.hpp"
        "threepp/lights/PointLight.hpp"

        "threepp/renderers/GLRenderer.hpp"

        "threepp/canvas.hpp"
        "threepp/constants.hpp"
        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
  list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

set(sources

        "threepp/Canvas.cpp"

        "threepp/core/BufferGeometry.cpp"
        "threepp/core/Object3D.cpp"

        "threepp/geometries/BoxGeometry.cpp"
        "threepp/geometries/CylinderGeometry.cpp"
        "threepp/geometries/PlaneGeometry.cpp"
        "threepp/geometries/SphereGeometry.cpp"

        "threepp/math/Box2.cpp"
        "threepp/math/Box3.cpp"
        "threepp/math/Color.cpp"
        "threepp/math/Cylindrical.cpp"
        "threepp/math/Euler.cpp"
        "threepp/math/Frustum.cpp"
        "threepp/math/Line3.cpp"
        "threepp/math/MathUtils.cpp"
        "threepp/math/Matrix3.cpp"
        "threepp/math/Matrix4.cpp"
        "threepp/math/Plane.cpp"
        "threepp/math/Ray.cpp"
        "threepp/math/Sphere.cpp"
        "threepp/math/Spherical.cpp"
        "threepp/math/SphericalHarmonics3.cpp"
        "threepp/math/Triangle.cpp"
        "threepp/math/Vector2.cpp"
        "threepp/math/Vector3.cpp"
        "threepp/math/Vector4.cpp"
        "threepp/math/Quaternion.cpp"

        "threepp/textures/Texture.cpp"

        "threepp/renderers/GLRenderer.cpp"

        )

set(privateHeaders

        "threepp/renderers/gl/GLClipping.hpp"
        "threepp/renderers/gl/GLInfo.hpp"
        "threepp/renderers/gl/GLProgram.hpp"

        "threepp/renderers/shaders/ShaderChunk.hpp"
        "threepp/renderers/shaders/ShaderLib.hpp"
        "threepp/renderers/shaders/UniformsLib.hpp"
        "threepp/renderers/shaders/UniformsUtil.hpp"

        )

add_library(threepp ${sources} ${privateHeaders} ${publicHeadersFull})
target_link_libraries(threepp PUBLIC CONAN_PKG::glfw CONAN_PKG::glad)
target_include_directories(threepp
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        private
        "${CMAKE_CURRENT_SOURCE_DIR}")

