set(publicHeaderDir ${CMAKE_SOURCE_DIR}/include)

set(publicHeaders

        "threepp/Canvas.hpp"
        "threepp/constants.hpp"
        "threepp/threepp.hpp"

        "threepp/controls/FlyControls.hpp"
        "threepp/controls/OrbitControls.hpp"

        "threepp/core/BufferAttribute.hpp"
        "threepp/core/BufferGeometry.hpp"
        "threepp/core/Clock.hpp"
        "threepp/core/EventDispatcher.hpp"
        "threepp/core/Face3.hpp"
        "threepp/core/Layers.hpp"
        "threepp/core/misc.hpp"
        "threepp/core/InstancedBufferAttribute.hpp"
        "threepp/core/InstancedBufferGeometry.hpp"
        "threepp/core/InterleavedBuffer.hpp"
        "threepp/core/InterleavedBufferAttribute.hpp"
        "threepp/core/Object3D.hpp"
        "threepp/core/Raycaster.hpp"
        "threepp/core/Shader.hpp"
        "threepp/core/Uniform.hpp"

        "threepp/cameras/Camera.hpp"
        "threepp/cameras/PerspectiveCamera.hpp"
        "threepp/cameras/OrthographicCamera.hpp"
        "threepp/cameras/CameraView.hpp"

        "threepp/extras/bullet/BulletWrapper.hpp"
        "threepp/extras/ShapeUtils.hpp"
        "threepp/extras/core/Curve.hpp"
        "threepp/extras/core/CurvePath.hpp"
        "threepp/extras/core/Interpolations.hpp"
        "threepp/extras/core/Path.hpp"
        "threepp/extras/core/Shape.hpp"
        "threepp/extras/core/ShapePath.hpp"
        "threepp/extras/curves/CatmullRomCurve3.hpp"
        "threepp/extras/curves/CubicBezierCurve.hpp"
        "threepp/extras/curves/CubicBezierCurve3.hpp"
        "threepp/extras/curves/EllipseCurve.hpp"
        "threepp/extras/curves/LineCurve.hpp"
        "threepp/extras/curves/SplineCurve.hpp"
        "threepp/extras/curves/QuadraticBezierCurve.hpp"
        "threepp/extras/curves/QuadraticBezierCurve3.hpp"
        "threepp/extras/imgui/imgui_context.hpp"

        "threepp/input/KeyListener.hpp"
        "threepp/input/MouseListener.hpp"

        "threepp/geometries/geometries.hpp"
        "threepp/geometries/BoxGeometry.hpp"
        "threepp/geometries/CapsuleGeometry.hpp"
        "threepp/geometries/CircleGeometry.hpp"
        "threepp/geometries/ConeGeometry.hpp"
        "threepp/geometries/ConvexGeometry.hpp"
        "threepp/geometries/CylinderGeometry.hpp"
        "threepp/geometries/DecalGeometry.hpp"
        "threepp/geometries/EdgesGeometry.hpp"
        "threepp/geometries/ExtrudeGeometry.hpp"
        "threepp/geometries/IcosahedronGeometry.hpp"
        "threepp/geometries/LatheGeometry.hpp"
        "threepp/geometries/OctahedronGeometry.hpp"
        "threepp/geometries/PlaneGeometry.hpp"
        "threepp/geometries/PolyhedronGeometry.hpp"
        "threepp/geometries/RingGeometry.hpp"
        "threepp/geometries/ShapeGeometry.hpp"
        "threepp/geometries/SphereGeometry.hpp"
        "threepp/geometries/TorusGeometry.hpp"
        "threepp/geometries/TorusKnotGeometry.hpp"
        "threepp/geometries/TubeGeometry.hpp"
        "threepp/geometries/WireframeGeometry.hpp"

        "threepp/helpers/helpers.hpp"
        "threepp/helpers/AxesHelper.hpp"
        "threepp/helpers/ArrowHelper.hpp"
        "threepp/helpers/Box3Helper.hpp"
        "threepp/helpers/CameraHelper.hpp"
        "threepp/helpers/DirectionalLightHelper.hpp"
        "threepp/helpers/GridHelper.hpp"
        "threepp/helpers/HemisphereLightHelper.hpp"
        "threepp/helpers/PlaneHelper.hpp"
        "threepp/helpers/PointLightHelper.hpp"
        "threepp/helpers/SpotLightHelper.hpp"

        "threepp/loaders/loaders.hpp"
        "threepp/loaders/AssimpLoader.hpp"
        "threepp/loaders/MTLLoader.hpp"
        "threepp/loaders/ImageLoader.hpp"
        "threepp/loaders/OBJLoader.hpp"
        "threepp/loaders/STLLoader.hpp"
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
        "threepp/materials/SpriteMaterial.hpp"
        "threepp/materials/interfaces.hpp"

        "threepp/math/Box2.hpp"
        "threepp/math/Box3.hpp"
        "threepp/math/Capsule.hpp"
        "threepp/math/Color.hpp"
        "threepp/math/Cylindrical.hpp"
        "threepp/math/Euler.hpp"
        "threepp/math/float_view.hpp"
        "threepp/math/Frustum.hpp"
        "threepp/math/ImprovedNoise.hpp"
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
        "threepp/objects/Sky.hpp"
        "threepp/objects/Sprite.hpp"
        "threepp/objects/Points.hpp"
        "threepp/objects/Reflector.hpp"
        "threepp/objects/Water.hpp"

        "threepp/textures/DepthTexture.hpp"
        "threepp/textures/Image.hpp"
        "threepp/textures/Texture.hpp"

        "threepp/utils/BufferGeometryUtils.hpp"
        "threepp/utils/ThreadPool.hpp"
        "threepp/utils/URLFetcher.hpp"

        "threepp/lights/lights.hpp"
        "threepp/lights/Light.hpp"
        "threepp/lights/LightProbe.hpp"
        "threepp/lights/AmbientLight.hpp"
        "threepp/lights/DirectionalLight.hpp"
        "threepp/lights/DirectionalLightShadow.hpp"
        "threepp/lights/HemisphereLight.hpp"
        "threepp/lights/LightShadow.hpp"
        "threepp/lights/PointLight.hpp"
        "threepp/lights/PointLightShadow.hpp"
        "threepp/lights/SpotLight.hpp"
        "threepp/lights/SpotLightShadow.hpp"

        "threepp/renderers/TextHandle.hpp"
        "threepp/renderers/GLRenderer.hpp"
        "threepp/renderers/GLRenderTarget.hpp"

        "threepp/renderers/gl/GLInfo.hpp"
        "threepp/renderers/gl/GLShadowMap.hpp"
        "threepp/renderers/gl/GLState.hpp"

        "threepp/renderers/shaders/ShaderChunk.hpp"
        "threepp/renderers/shaders/ShaderLib.hpp"
        "threepp/renderers/shaders/UniformsLib.hpp"

        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
    list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

set(privateHeaders

        "threepp/extras/earcut.hpp"
        "threepp/extras/quickhull.hpp"

        "threepp/materials/MeshDistanceMaterial.hpp"

        "threepp/renderers/gl/Buffer.hpp"
        "threepp/renderers/gl/GLAttributes.hpp"
        "threepp/renderers/gl/GLBackground.hpp"
        "threepp/renderers/gl/GLBindingStates.hpp"
        "threepp/renderers/gl/GLBufferRenderer.hpp"
        "threepp/renderers/gl/GLCapabilities.hpp"
        "threepp/renderers/gl/GLClipping.hpp"
        "threepp/renderers/gl/GLGeometries.hpp"
        "threepp/renderers/gl/GLLights.hpp"
        "threepp/renderers/gl/GLMaterials.hpp"
        "threepp/renderers/gl/GLObjects.hpp"
        "threepp/renderers/gl/GLProperties.hpp"
        "threepp/renderers/gl/GLProgram.hpp"
        "threepp/renderers/gl/GLPrograms.hpp"
        "threepp/renderers/gl/GLRenderLists.hpp"
        "threepp/renderers/gl/GLRenderStates.hpp"
        "threepp/renderers/gl/GLTextures.hpp"
        "threepp/renderers/gl/GLUniforms.hpp"
        "threepp/renderers/gl/GLUtils.hpp"
        "threepp/renderers/gl/UniformUtils.hpp"

        "threepp/utils/regex_util.hpp"
        "threepp/utils/StringUtils.hpp"

        )

set(sources

        "threepp/Canvas.cpp"

        "threepp/cameras/PerspectiveCamera.cpp"
        "threepp/cameras/OrthographicCamera.cpp"

        "threepp/controls/FlyControls.cpp"
        "threepp/controls/OrbitControls.cpp"

        "threepp/core/BufferGeometry.cpp"
        "threepp/core/Clock.cpp"
        "threepp/core/EventDispatcher.cpp"
        "threepp/core/Layers.cpp"
        "threepp/core/Object3D.cpp"
        "threepp/core/Raycaster.cpp"
        "threepp/core/Uniform.cpp"

        "threepp/extras/ShapeUtils.cpp"
        "threepp/extras/core/Curve.cpp"
        "threepp/extras/core/CurvePath.cpp"
        "threepp/extras/core/Path.cpp"
        "threepp/extras/core/Shape.cpp"
        "threepp/extras/core/ShapePath.cpp"
        "threepp/extras/curves/CatmullRomCurve3.cpp"
        "threepp/extras/curves/CubicBezierCurve.cpp"
        "threepp/extras/curves/CubicBezierCurve3.cpp"
        "threepp/extras/curves/EllipseCurve.cpp"
        "threepp/extras/curves/LineCurve.cpp"
        "threepp/extras/curves/QuadraticBezierCurve.cpp"
        "threepp/extras/curves/QuadraticBezierCurve3.cpp"
        "threepp/extras/curves/SplineCurve.cpp"

        "threepp/geometries/BoxGeometry.cpp"
        "threepp/geometries/CapsuleGeometry.cpp"
        "threepp/geometries/CircleGeometry.cpp"
        "threepp/geometries/ConvexGeometry.cpp"
        "threepp/geometries/CylinderGeometry.cpp"
        "threepp/geometries/DecalGeometry.cpp"
        "threepp/geometries/EdgesGeometry.cpp"
        "threepp/geometries/ExtrudeGeometry.cpp"
        "threepp/geometries/IcosahedronGeometry.cpp"
        "threepp/geometries/LatheGeometry.cpp"
        "threepp/geometries/OctahedronGeometry.cpp"
        "threepp/geometries/PlaneGeometry.cpp"
        "threepp/geometries/PolyhedronGeometry.cpp"
        "threepp/geometries/RingGeometry.cpp"
        "threepp/geometries/ShapeGeometry.cpp"
        "threepp/geometries/SphereGeometry.cpp"
        "threepp/geometries/TorusGeometry.cpp"
        "threepp/geometries/TorusKnotGeometry.cpp"
        "threepp/geometries/TubeGeometry.cpp"
        "threepp/geometries/WireframeGeometry.cpp"

        "threepp/helpers/ArrowHelper.cpp"
        "threepp/helpers/AxesHelper.cpp"
        "threepp/helpers/Box3Helper.cpp"
        "threepp/helpers/CameraHelper.cpp"
        "threepp/helpers/DirectionalLightHelper.cpp"
        "threepp/helpers/GridHelper.cpp"
        "threepp/helpers/HemisphereLightHelper.cpp"
        "threepp/helpers/PlaneHelper.cpp"
        "threepp/helpers/PointLightHelper.cpp"
        "threepp/helpers/SpotLightHelper.cpp"

        "threepp/loaders/ImageLoader.cpp"
        "threepp/loaders/MTLLoader.cpp"
        "threepp/loaders/OBJLoader.cpp"
        "threepp/loaders/STLLoader.cpp"
        "threepp/loaders/TextureLoader.cpp"

        "threepp/materials/LineBasicMaterial.cpp"
        "threepp/materials/Material.cpp"
        "threepp/materials/MeshBasicMaterial.cpp"
        "threepp/materials/MeshLambertMaterial.cpp"
        "threepp/materials/MeshNormalMaterial.cpp"
        "threepp/materials/MeshPhongMaterial.cpp"
        "threepp/materials/MeshStandardMaterial.cpp"
        "threepp/materials/PointsMaterial.cpp"
        "threepp/materials/SpriteMaterial.cpp"

        "threepp/math/Box2.cpp"
        "threepp/math/Box3.cpp"
        "threepp/math/Capsule.cpp"
        "threepp/math/Color.cpp"
        "threepp/math/Cylindrical.cpp"
        "threepp/math/Euler.cpp"
        "threepp/math/Frustum.cpp"
        "threepp/math/ImprovedNoise.cpp"
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
        "threepp/lights/SpotLight.cpp"
        "threepp/lights/SpotLightShadow.cpp"

        "threepp/scenes/Scene.cpp"
        "threepp/scenes/Fog.cpp"
        "threepp/scenes/FogExp2.cpp"

        "threepp/objects/Group.cpp"
        "threepp/objects/Line.cpp"
        "threepp/objects/LineLoop.cpp"
        "threepp/objects/LineSegments.cpp"
        "threepp/objects/LOD.cpp"
        "threepp/objects/InstancedMesh.cpp"
        "threepp/objects/Mesh.cpp"
        "threepp/objects/Points.cpp"
        "threepp/objects/Sky.cpp"
        "threepp/objects/Sprite.cpp"
        "threepp/objects/Reflector.cpp"
        "threepp/objects/Water.cpp"

        "threepp/textures/Texture.cpp"
        "threepp/textures/DataTexture3D.cpp"

        "threepp/utils/BufferGeometryUtils.cpp"
        "threepp/utils/ThreadPool.cpp"

        "threepp/renderers/TextHandle.cpp"
        "threepp/renderers/GLRenderer.cpp"
        "threepp/renderers/GLRenderTarget.cpp"

        "threepp/renderers/gl/GLAttributes.cpp"
        "threepp/renderers/gl/GLBackground.cpp"
        "threepp/renderers/gl/GLBindingStates.cpp"
        "threepp/renderers/gl/GLBufferRenderer.cpp"
        "threepp/renderers/gl/GLCapabilities.cpp"
        "threepp/renderers/gl/GLClipping.cpp"
        "threepp/renderers/gl/GLGeometries.cpp"
        "threepp/renderers/gl/GLInfo.cpp"
        "threepp/renderers/gl/GLLights.cpp"
        "threepp/renderers/gl/GLObjects.cpp"
        "threepp/renderers/gl/GLProgram.cpp"
        "threepp/renderers/gl/GLPrograms.cpp"
        "threepp/renderers/gl/GLMaterials.cpp"
        "threepp/renderers/gl/GLRenderLists.cpp"
        "threepp/renderers/gl/GLRenderStates.cpp"
        "threepp/renderers/gl/GLShadowMap.cpp"
        "threepp/renderers/gl/GLState.cpp"
        "threepp/renderers/gl/GLTextures.cpp"
        "threepp/renderers/gl/GLUniforms.cpp"
        "threepp/renderers/gl/ProgramParameters.cpp"

        "threepp/renderers/shaders/ShaderLib.cpp"

        )

if (nlohmann_json_FOUND)
    list(APPEND sources "threepp/loaders/FontLoader.cpp" "threepp/extras/core/Font.cpp")
endif()

if (CURL_FOUND)
    list(APPEND sources "threepp/utils/URLFetcher.cpp")
endif()

include("${CMAKE_SOURCE_DIR}/cmake/shaders.cmake")

add_library(threepp ${sources} ${privateHeaders} ${publicHeadersFull} "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp")
add_library(threepp::threepp ALIAS threepp)
target_compile_features(threepp PUBLIC cxx_std_17)
target_link_libraries(threepp
        PRIVATE
        glad::glad
        glfw::glfw)
if (nlohmann_json_FOUND)
    target_link_libraries(threepp PRIVATE nlohmann_json::nlohmann_json)
endif()
if (CURL_FOUND)
    target_link_libraries(threepp PRIVATE CURL::libcurl)
    target_compile_definitions(threepp PUBLIC "THREEPP_WITH_CURL")
endif()
if(UNIX)
    target_link_libraries(threepp PRIVATE pthread)
endif()
target_include_directories(threepp
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        "${stb_INCLUDE_DIR}"
        "${gltext_INCLUDE_DIR}"
        "${generatedSourcesDir}"
        "${CMAKE_CURRENT_SOURCE_DIR}")
