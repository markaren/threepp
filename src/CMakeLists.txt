set(publicHeaderDir ${PROJECT_SOURCE_DIR}/include)

set(publicHeaders

        "threepp/constants.hpp"
        "threepp/threepp.hpp"

        "threepp/canvas/Canvas.hpp"
        "threepp/canvas/Monitor.hpp"
        "threepp/canvas/WindowSize.hpp"

        "threepp/controls/DragControls.hpp"
        "threepp/controls/FlyControls.hpp"
        "threepp/controls/OrbitControls.hpp"

        "threepp/core/BufferAttribute.hpp"
        "threepp/core/BufferGeometry.hpp"
        "threepp/core/Clock.hpp"
        "threepp/core/EventDispatcher.hpp"
        "threepp/core/Face3.hpp"
        "threepp/core/Layers.hpp"
        "threepp/core/misc.hpp"
        "threepp/core/InterleavedBuffer.hpp"
        "threepp/core/InterleavedBufferAttribute.hpp"
        "threepp/core/Object3D.hpp"
        "threepp/core/Raycaster.hpp"
        "threepp/core/Shader.hpp"
        "threepp/core/Uniform.hpp"

        "threepp/cameras/Camera.hpp"
        "threepp/cameras/CubeCamera.hpp"
        "threepp/cameras/PerspectiveCamera.hpp"
        "threepp/cameras/OrthographicCamera.hpp"

        "threepp/extras/ShapeUtils.hpp"
        "threepp/extras/core/Curve.hpp"
        "threepp/extras/core/CurvePath.hpp"
        "threepp/extras/core/Font.hpp"
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
        "threepp/extras/imgui/ImguiContext.hpp"

        "threepp/input/KeyListener.hpp"
        "threepp/input/MouseListener.hpp"
        "threepp/input/PeripheralsEventSource.hpp"

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
        "threepp/geometries/ExtrudeTextGeometry.hpp"
        "threepp/geometries/IcosahedronGeometry.hpp"
        "threepp/geometries/LatheGeometry.hpp"
        "threepp/geometries/OctahedronGeometry.hpp"
        "threepp/geometries/PlaneGeometry.hpp"
        "threepp/geometries/PolyhedronGeometry.hpp"
        "threepp/geometries/RingGeometry.hpp"
        "threepp/geometries/ShapeGeometry.hpp"
        "threepp/geometries/SphereGeometry.hpp"
        "threepp/geometries/TextGeometry.hpp"
        "threepp/geometries/TorusGeometry.hpp"
        "threepp/geometries/TorusKnotGeometry.hpp"
        "threepp/geometries/TubeGeometry.hpp"
        "threepp/geometries/WireframeGeometry.hpp"

        "threepp/helpers/helpers.hpp"
        "threepp/helpers/AxesHelper.hpp"
        "threepp/helpers/ArrowHelper.hpp"
        "threepp/helpers/Box3Helper.hpp"
        "threepp/helpers/BoxHelper.hpp"
        "threepp/helpers/CameraHelper.hpp"
        "threepp/helpers/DirectionalLightHelper.hpp"
        "threepp/helpers/GridHelper.hpp"
        "threepp/helpers/HemisphereLightHelper.hpp"
        "threepp/helpers/PlaneHelper.hpp"
        "threepp/helpers/PointLightHelper.hpp"
        "threepp/helpers/PolarGridHelper.hpp"
        "threepp/helpers/SkeletonHelper.hpp"
        "threepp/helpers/SpotLightHelper.hpp"

        "threepp/loaders/loaders.hpp"
        "threepp/loaders/AssimpLoader.hpp"
        "threepp/loaders/CubeTextureLoader.hpp"
        "threepp/loaders/FontLoader.hpp"
        "threepp/loaders/MTLLoader.hpp"
        "threepp/loaders/ImageLoader.hpp"
        "threepp/loaders/OBJLoader.hpp"
        "threepp/loaders/STLLoader.hpp"
        "threepp/loaders/TextureLoader.hpp"
        "threepp/loaders/URDFLoader.hpp"

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
        "threepp/math/Lut.hpp"
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

        "threepp/objects/Bone.hpp"
        "threepp/objects/Group.hpp"
        "threepp/objects/HUD.hpp"
        "threepp/objects/InstancedMesh.hpp"
        "threepp/objects/Line.hpp"
        "threepp/objects/LineLoop.hpp"
        "threepp/objects/LineSegments.hpp"
        "threepp/objects/LOD.hpp"
        "threepp/objects/Mesh.hpp"
        "threepp/objects/ObjectWithMaterials.hpp"
        "threepp/objects/ObjectWithMorphTargetInfluences.hpp"
        "threepp/objects/ParticleSystem.hpp"
        "threepp/objects/Sky.hpp"
        "threepp/objects/Skeleton.hpp"
        "threepp/objects/SkinnedMesh.hpp"
        "threepp/objects/Sprite.hpp"
        "threepp/objects/Points.hpp"
        "threepp/objects/Reflector.hpp"
        "threepp/objects/Text.hpp"
        "threepp/objects/Water.hpp"

        "threepp/textures/CubeTexture.hpp"
        "threepp/textures/DataTexture.hpp"
        "threepp/textures/DataTexture3D.hpp"
        "threepp/textures/DepthTexture.hpp"
        "threepp/textures/Image.hpp"
        "threepp/textures/Texture.hpp"

        "threepp/utils/BufferGeometryUtils.hpp"
        "threepp/utils/ImageUtils.hpp"
        "threepp/utils/StringUtils.hpp"
        "threepp/utils/TaskManager.hpp"

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

        "threepp/materials/MeshDistanceMaterial.hpp"

        "threepp/renderers/GLCubeRenderTarget.hpp"

        "threepp/renderers/gl/Buffer.hpp"
        "threepp/renderers/gl/GLAttributes.hpp"
        "threepp/renderers/gl/GLBackground.hpp"
        "threepp/renderers/gl/GLBindingStates.hpp"
        "threepp/renderers/gl/GLBufferRenderer.hpp"
        "threepp/renderers/gl/GLCapabilities.hpp"
        "threepp/renderers/gl/GLCubeMaps.hpp"
        "threepp/renderers/gl/GLClipping.hpp"
        "threepp/renderers/gl/GLGeometries.hpp"
        "threepp/renderers/gl/GLLights.hpp"
        "threepp/renderers/gl/GLMaterials.hpp"
        "threepp/renderers/gl/GLMorphTargets.hpp"
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

        "threepp/utils/RegexUtil.hpp"
)

set(sources

        "threepp/cameras/Camera.cpp"
        "threepp/cameras/PerspectiveCamera.cpp"
        "threepp/cameras/OrthographicCamera.cpp"

        "threepp/controls/DragControls.cpp"
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
        "threepp/extras/core/Font.cpp"
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
        "threepp/geometries/ConeGeometry.cpp"
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
        "threepp/geometries/TextGeometry.cpp"
        "threepp/geometries/TorusGeometry.cpp"
        "threepp/geometries/TorusKnotGeometry.cpp"
        "threepp/geometries/TubeGeometry.cpp"
        "threepp/geometries/WireframeGeometry.cpp"

        "threepp/helpers/ArrowHelper.cpp"
        "threepp/helpers/AxesHelper.cpp"
        "threepp/helpers/Box3Helper.cpp"
        "threepp/helpers/BoxHelper.cpp"
        "threepp/helpers/CameraHelper.cpp"
        "threepp/helpers/DirectionalLightHelper.cpp"
        "threepp/helpers/GridHelper.cpp"
        "threepp/helpers/HemisphereLightHelper.cpp"
        "threepp/helpers/PlaneHelper.cpp"
        "threepp/helpers/PointLightHelper.cpp"
        "threepp/helpers/PolarGridHelper.cpp"
        "threepp/helpers/SkeletonHelper.cpp"
        "threepp/helpers/SpotLightHelper.cpp"

        "threepp/input/PeripheralsEventSource.cpp"

        "threepp/loaders/FontLoader.cpp"
        "threepp/loaders/ImageLoader.cpp"
        "threepp/loaders/MTLLoader.cpp"
        "threepp/loaders/OBJLoader.cpp"
        "threepp/loaders/STLLoader.cpp"
        "threepp/loaders/TextureLoader.cpp"
        "threepp/loaders/URDFLoader.cpp"

        "threepp/materials/LineBasicMaterial.cpp"
        "threepp/materials/Material.cpp"
        "threepp/materials/MeshBasicMaterial.cpp"
        "threepp/materials/MeshLambertMaterial.cpp"
        "threepp/materials/MeshNormalMaterial.cpp"
        "threepp/materials/MeshPhongMaterial.cpp"
        "threepp/materials/MeshStandardMaterial.cpp"
        "threepp/materials/PointsMaterial.cpp"
        "threepp/materials/RawShaderMaterial.cpp"
        "threepp/materials/ShaderMaterial.cpp"
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

        "threepp/lights/AmbientLight.cpp"
        "threepp/lights/DirectionalLight.cpp"
        "threepp/lights/HemisphereLight.cpp"
        "threepp/lights/Light.cpp"
        "threepp/lights/LightShadow.cpp"
        "threepp/lights/PointLight.cpp"
        "threepp/lights/PointLightShadow.cpp"
        "threepp/lights/SpotLight.cpp"
        "threepp/lights/SpotLightShadow.cpp"

        "threepp/scenes/Scene.cpp"
        "threepp/scenes/Fog.cpp"
        "threepp/scenes/FogExp2.cpp"

        "threepp/objects/Group.cpp"
        "threepp/objects/HUD.cpp"
        "threepp/objects/Line.cpp"
        "threepp/objects/LineLoop.cpp"
        "threepp/objects/LineSegments.cpp"
        "threepp/objects/LOD.cpp"
        "threepp/objects/InstancedMesh.cpp"
        "threepp/objects/Mesh.cpp"
        "threepp/objects/ObjectWithMaterials.cpp"
        "threepp/objects/ParticleSystem.cpp"
        "threepp/objects/Points.cpp"
        "threepp/objects/Skeleton.cpp"
        "threepp/objects/SkinnedMesh.cpp"
        "threepp/objects/Sky.cpp"
        "threepp/objects/Sprite.cpp"
        "threepp/objects/Reflector.cpp"
        "threepp/objects/Water.cpp"

        "threepp/textures/Texture.cpp"
        "threepp/textures/DataTexture3D.cpp"

        "threepp/utils/BufferGeometryUtils.cpp"
        "threepp/utils/StringUtils.cpp"
        "threepp/utils/TaskManager.cpp"

        "threepp/renderers/GLRenderer.cpp"
        "threepp/renderers/GLRenderTarget.cpp"

        "threepp/renderers/gl/GLAttributes.cpp"
        "threepp/renderers/gl/GLBackground.cpp"
        "threepp/renderers/gl/GLBindingStates.cpp"
        "threepp/renderers/gl/GLBufferRenderer.cpp"
        "threepp/renderers/gl/GLClipping.cpp"
        "threepp/renderers/gl/GLCubeMaps.cpp"
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

if (THREEPP_WITH_SVG)
    list(APPEND publicHeaders
            "threepp/loaders/SVGLoader.hpp"
    )

    list(APPEND sources
            "threepp/loaders/SVGLoader.cpp"
            "external/pugixml/pugixml.cpp"
    )
endif ()

if (THREEPP_WITH_AUDIO)
    list(APPEND publicHeaders
            "threepp/audio/Audio.hpp"
    )

    list(APPEND sources
            "threepp/audio/Audio.cpp"
    )
endif ()

if (THREEPP_WITH_GLFW)
    list(APPEND publicHeaders
            "threepp/canvas/Canvas.hpp"
    )

    list(APPEND sources
            "threepp/canvas/Canvas.cpp"
    )
    if (NOT EMSCRIPTEN AND NOT THREEPP_USE_EXTERNAL_GLFW)

        message("Using threepp's GLFW")

        #https://github.com/raysan5/raylib/blob/master/cmake/GlfwImport.cmake
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_COCOA OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
        if (THREEPP_EMBEDDED_GLFW_WITH_X11)
            set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
        else ()
            set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
        endif ()
        set(GLFW_LIBRARY_TYPE "OBJECT" CACHE STRING "" FORCE)
        add_subdirectory("external/glfw")

        # Hide glfw's symbols when building a shared lib
        if (BUILD_SHARED_LIBS)
            set_property(TARGET glfw PROPERTY C_VISIBILITY_PRESET hidden)
        endif ()

        list(APPEND sources $<TARGET_OBJECTS:glfw>)
    endif ()
endif ()

if (NOT DEFINED EMSCRIPTEN)
    list(APPEND privateHeaders
            "threepp/utils/LoadGlad.hpp"
    )
    list(APPEND sources
            "external/glad/glad.c"
            "threepp/utils/LoadGlad.cpp"
    )
endif ()

include("${PROJECT_SOURCE_DIR}/cmake/embed.cmake")

list(APPEND sources
        "${generatedSourcesDir}/threepp/EmbeddedFonts.cpp"
        "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp"
)

add_library(threepp ${sources} ${privateHeaders} ${publicHeadersFull})
add_library(threepp::threepp ALIAS threepp)
target_compile_features(threepp PUBLIC "cxx_std_20")

if (UNIX)
    target_link_libraries(threepp PRIVATE pthread dl)
endif ()
target_include_directories(threepp
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        "${generatedSourcesDir}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/earcut"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/miniaudio"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/pugixml"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/quickhull"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
)

if (NOT DEFINED EMSCRIPTEN)

    target_include_directories(threepp
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/external/glad"
    )

    if (THREEPP_USE_EXTERNAL_GLFW)
        target_link_libraries(threepp PRIVATE glfw::glfw)
    else ()
        target_include_directories(threepp PRIVATE BEFORE SYSTEM "external/glfw/include")
    endif ()

endif ()


if (THREEPP_TREAT_WARNINGS_AS_ERRORS)
    # Treat warnings as errors,
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(threepp PRIVATE "-Wall" "-Wextra")
        target_compile_options(threepp PRIVATE
                "-Wno-parentheses" "-Wno-reorder" "-Wno-reorder-ctor"
                "-Wno-missing-field-initializers" "-Wno-overloaded-virtual"
                "-Wno-unused-parameter" "-Wno-dollar-in-identifier-extension")
        target_compile_options(threepp PRIVATE "-Werror")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(threepp PRIVATE "/WX")
    endif ()
endif ()
