set(publicHeaderDir ${CMAKE_SOURCE_DIR}/include)

set(publicHeaders

        "threepp/Canvas.hpp"
        "threepp/constants.hpp"
        "threepp/threepp.hpp"

        "threepp/controls/OrbitControls.hpp"

        "threepp/core/BufferAttribute.hpp"
        "threepp/core/BufferGeometry.hpp"
        "threepp/core/Clock.hpp"
        "threepp/core/Face3.hpp"
        "threepp/core/Layers.hpp"
        "threepp/core/misc.hpp"
        "threepp/core/InstancedBufferAttribute.hpp"
        "threepp/core/InstancedBufferGeometry.hpp"
        "threepp/core/Object3D.hpp"
        "threepp/core/Raycaster.hpp"
        "threepp/core/Uniform.hpp"

        "threepp/cameras/Camera.hpp"
        "threepp/cameras/PerspectiveCamera.hpp"
        "threepp/cameras/OrthographicCamera.hpp"
        "threepp/cameras/CameraView.hpp"

        "threepp/input/KeyListener.hpp"
        "threepp/input/MouseListener.hpp"

        "threepp/geometries/geometries.hpp"
        "threepp/geometries/BoxGeometry.hpp"
        "threepp/geometries/CircleGeometry.hpp"
        "threepp/geometries/ConeGeometry.hpp"
        "threepp/geometries/CylinderGeometry.hpp"
        "threepp/geometries/PlaneGeometry.hpp"
        "threepp/geometries/SphereGeometry.hpp"

        "threepp/helpers/helpers.hpp"
        "threepp/helpers/AxesHelper.hpp"
        "threepp/helpers/ArrowHelper.hpp"
        "threepp/helpers/Box3Helper.hpp"
        "threepp/helpers/DirectionalLightHelper.hpp"
        "threepp/helpers/GridHelper.hpp"
        "threepp/helpers/PlaneHelper.hpp"

        "threepp/loaders/ImageLoader.hpp"
        "threepp/loaders/TextureLoader.hpp"

        "threepp/materials/Material.hpp"
        "threepp/materials/MeshBasicMaterial.hpp"
        "threepp/materials/MeshDepthMaterial.hpp"
        "threepp/materials/MeshLambertMaterial.hpp"
        "threepp/materials/MeshNormalMaterial.hpp"
        "threepp/materials/MeshPhongMaterial.hpp"
        "threepp/materials/MeshStandardMaterial.hpp"
        "threepp/materials/MeshToonMaterial.hpp"
        "threepp/materials/PointsMaterial.hpp"
        "threepp/materials/RawShaderMaterial.hpp"
        "threepp/materials/ShaderMaterial.hpp"
        "threepp/materials/ShadowMaterial.hpp"
        "threepp/materials/interfaces.hpp"

        "threepp/math/Box2.hpp"
        "threepp/math/Box3.hpp"
        "threepp/math/Color.hpp"
        "threepp/math/Cylindrical.hpp"
        "threepp/math/Euler.hpp"
        "threepp/math/float_view.hpp"
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

        "threepp/objects/Group.hpp"
        "threepp/objects/InstancedMesh.hpp"
        "threepp/objects/Line.hpp"
        "threepp/objects/LineLoop.hpp"
        "threepp/objects/LineSegments.hpp"
        "threepp/objects/LOD.hpp"
        "threepp/objects/Mesh.hpp"
        "threepp/objects/Points.hpp"
        
        "threepp/textures/Image.hpp"
        "threepp/textures/Texture.hpp"

        "threepp/utils/uuid.hpp"

        "threepp/lights/lights.hpp"
        "threepp/lights/Light.hpp"
        "threepp/lights/LightProbe.hpp"
        "threepp/lights/AmbientLight.hpp"
        "threepp/lights/DirectionalLight.hpp"
        "threepp/lights/DirectionalLightShadow.hpp"
        "threepp/lights/LightShadow.hpp"
        "threepp/lights/PointLight.hpp"
        "threepp/lights/PointLightShadow.hpp"
        "threepp/lights/SpotLight.hpp"

        "threepp/renderers/GLRenderer.hpp"
        "threepp/renderers/GLRenderTarget.hpp"

        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
  list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

set(privateHeaders

        "threepp/renderers/gl/GLAttributes.hpp"
        "threepp/renderers/gl/GLBackground.hpp"
        "threepp/renderers/gl/GLBindingStates.hpp"
        "threepp/renderers/gl/GLBufferRenderer.hpp"
        "threepp/renderers/gl/GLCapabilities.hpp"
        "threepp/renderers/gl/GLClipping.hpp"
        "threepp/renderers/gl/GLGeometries.hpp"
        "threepp/renderers/gl/GLInfo.hpp"
        "threepp/renderers/gl/GLLights.hpp"
        "threepp/renderers/gl/GLMaterials.hpp"
        "threepp/renderers/gl/GLObjects.hpp"
        "threepp/renderers/gl/glHelper.hpp"
        "threepp/renderers/gl/GLProperties.hpp"
        "threepp/renderers/gl/GLProgram.hpp"
        "threepp/renderers/gl/GLPrograms.hpp"
        "threepp/renderers/gl/GLRenderLists.hpp"
        "threepp/renderers/gl/GLRenderStates.hpp"
        "threepp/renderers/gl/GLShadowMap.hpp"
        "threepp/renderers/gl/GLState.hpp"
        "threepp/renderers/gl/GLTextures.hpp"
        "threepp/renderers/gl/GLUniforms.hpp"
        "threepp/renderers/gl/GLUtils.hpp"
        "threepp/renderers/gl/ProgramParameters.hpp"

        "threepp/renderers/shaders/Shader.hpp"
        "threepp/renderers/shaders/ShaderChunk.hpp"
        "threepp/renderers/shaders/ShaderLib.hpp"
        "threepp/renderers/shaders/UniformsLib.hpp"
        "threepp/renderers/shaders/UniformsUtil.hpp"

        "threepp/utils/InstanceOf.hpp"
        "threepp/utils/regex_util.hpp"
        "threepp/utils/StringUtils.hpp"

        )

set(sources

        "threepp/Canvas.cpp"

        "threepp/cameras/PerspectiveCamera.cpp"
        "threepp/cameras/OrthographicCamera.cpp"

        "threepp/controls/OrbitControls.cpp"

        "threepp/core/BufferGeometry.cpp"
        "threepp/core/Object3D.cpp"
        "threepp/core/Raycaster.cpp"

        "threepp/geometries/BoxGeometry.cpp"
        "threepp/geometries/CylinderGeometry.cpp"
        "threepp/geometries/PlaneGeometry.cpp"
        "threepp/geometries/SphereGeometry.cpp"

        "threepp/helpers/ArrowHelper.cpp"

        "threepp/loaders/ImageLoader.cpp"

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

        "threepp/lights/PointLight.cpp"
        "threepp/lights/PointLightShadow.cpp"

        "threepp/objects/Mesh.cpp"

        "threepp/textures/Texture.cpp"

        "threepp/utils/uuid.cpp"

        "threepp/renderers/GLRenderer.cpp"

        "threepp/renderers/gl/GLAttributes.cpp"
        "threepp/renderers/gl/GLBackground.cpp"
        "threepp/renderers/gl/GLBindingStates.cpp"
        "threepp/renderers/gl/GLBufferRenderer.cpp"
        "threepp/renderers/gl/GLGeometries.cpp"
        "threepp/renderers/gl/GLInfo.cpp"
        "threepp/renderers/gl/GLLights.cpp"
        "threepp/renderers/gl/GLProgram.cpp"
        "threepp/renderers/gl/GLPrograms.cpp"
        "threepp/renderers/gl/GLMaterials.cpp"
        "threepp/renderers/gl/GLShadowMap.cpp"
        "threepp/renderers/gl/GLState.cpp"
        "threepp/renderers/gl/GLTextures.cpp"
        "threepp/renderers/gl/GLUniforms.cpp"
        "threepp/renderers/gl/GLRenderLists.cpp"
        "threepp/renderers/gl/ProgramParameters.cpp"

        )

add_library(threepp ${sources} ${privateHeaders} ${publicHeadersFull})
target_link_libraries(threepp
        PUBLIC
        CONAN_PKG::glfw
        CONAN_PKG::glad
        PRIVATE
        CONAN_PKG::stb)
target_include_directories(threepp
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
        private
        "${CMAKE_CURRENT_SOURCE_DIR}")

