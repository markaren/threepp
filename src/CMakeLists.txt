set(publicHeaderDir ${PROJECT_SOURCE_DIR}/include)

set(publicHeaders

        "threepp/constants.hpp"
        "threepp/threepp.hpp"

        "threepp/animation/AnimationAction.hpp"
        "threepp/animation/AnimationClip.hpp"
        "threepp/animation/AnimationMixer.hpp"
        "threepp/animation/AnimationUtils.hpp"
        "threepp/animation/KeyframeTrack.hpp"

        "threepp/animation/tracks/VectorKeyframeTrack.hpp"
        "threepp/animation/tracks/QuaternionKeyframeTrack.hpp"

        "threepp/canvas/Canvas.hpp"
        "threepp/canvas/Monitor.hpp"
        "threepp/canvas/WindowSize.hpp"

        "threepp/controls/DragControls.hpp"
        "threepp/controls/FlyControls.hpp"
        "threepp/controls/OrbitControls.hpp"
        "threepp/controls/TransformControls.hpp"

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
        "threepp/helpers/DepthSensor.hpp"
        "threepp/helpers/LidarModel.hpp"
        "threepp/helpers/LidarSensor.hpp"
        "threepp/helpers/DirectionalLightHelper.hpp"
        "threepp/helpers/GridHelper.hpp"
        "threepp/helpers/HemisphereLightHelper.hpp"
        "threepp/helpers/PlaneHelper.hpp"
        "threepp/helpers/PointLightHelper.hpp"
        "threepp/helpers/PolarGridHelper.hpp"
        "threepp/helpers/SkeletonHelper.hpp"
        "threepp/helpers/SpotLightHelper.hpp"

        "threepp/loaders/AsyncGroup.hpp"
        "threepp/loaders/loaders.hpp"
        "threepp/loaders/AssimpLoader.hpp"
        "threepp/loaders/CubeTextureLoader.hpp"
        "threepp/loaders/ColladaLoader.hpp"
        "threepp/loaders/DDSLoader.hpp"
        "threepp/loaders/FontLoader.hpp"
        "threepp/loaders/GLTFLoader.hpp"
        "threepp/loaders/MaterialVariants.hpp"
        "threepp/loaders/MTLLoader.hpp"
        "threepp/loaders/ImageLoader.hpp"
        "threepp/loaders/RGBELoader.hpp"
        "threepp/loaders/ModelLoader.hpp"
        "threepp/loaders/OBJLoader.hpp"
        "threepp/loaders/STLLoader.hpp"
        "threepp/loaders/TextureLoader.hpp"
        "threepp/loaders/URDFLoader.hpp"

        "threepp/materials/LineDashedMaterial.hpp"
        "threepp/materials/Material.hpp"
        "threepp/materials/MeshBasicMaterial.hpp"
        "threepp/materials/MeshDepthMaterial.hpp"
        "threepp/materials/MeshLambertMaterial.hpp"
        "threepp/materials/MeshNormalMaterial.hpp"
        "threepp/materials/MeshPhongMaterial.hpp"
        "threepp/materials/MeshPhysicalMaterial.hpp"
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
        "threepp/math/Interpolant.hpp"
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
        "threepp/objects/TextSprite.hpp"
        "threepp/objects/Water.hpp"
        "threepp/objects/DisplacedMesh.hpp"

        "threepp/textures/CubeTexture.hpp"
        "threepp/textures/DataTexture.hpp"
        "threepp/textures/DataTexture3D.hpp"
        "threepp/textures/DepthTexture.hpp"
        "threepp/textures/FramebufferTexture.hpp"
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
        "threepp/lights/RectAreaLight.hpp"
        "threepp/lights/RectAreaLightUniformsLib.hpp"
        "threepp/lights/SpotLight.hpp"
        "threepp/lights/SpotLightShadow.hpp"

        "threepp/renderers/Renderer.hpp"
        "threepp/renderers/RendererFactory.hpp"
        "threepp/renderers/GLRenderer.hpp"
        "threepp/renderers/WgpuRenderer.hpp"
        "threepp/renderers/CrossRenderer.hpp"
        "threepp/renderers/RenderTarget.hpp"
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

        "threepp/animation/PropertyBinding.hpp"
        "threepp/animation/PropertyMixer.hpp"

        "threepp/materials/MeshDistanceMaterial.hpp"

        "threepp/renderers/GLCubeRenderTarget.hpp"

        "threepp/renderers/common/Lights.hpp"
        "threepp/renderers/common/RenderLists.hpp"
        "threepp/renderers/common/RenderStates.hpp"
        "threepp/renderers/common/RendererCapabilities.hpp"
        "threepp/renderers/common/ShadowConfig.hpp"

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
        "threepp/renderers/gl/GLPMREM.hpp"
        "threepp/renderers/gl/GLProperties.hpp"
        "threepp/renderers/gl/GLProgram.hpp"
        "threepp/renderers/gl/GLPrograms.hpp"
        "threepp/renderers/gl/GLRenderLists.hpp"
        "threepp/renderers/gl/GLRenderStates.hpp"
        "threepp/renderers/gl/GLTextures.hpp"
        "threepp/renderers/gl/GLUniforms.hpp"
        "threepp/renderers/gl/GLUtils.hpp"
        "threepp/renderers/gl/ProgramParameters.hpp"
        "threepp/renderers/gl/UniformUtils.hpp"

        "threepp/utils/RegexUtil.hpp"
)

set(sources

        "external/pugixml/pugixml.cpp"

        "threepp/animation/AnimationAction.cpp"
        "threepp/animation/AnimationClip.cpp"
        "threepp/animation/AnimationMixer.cpp"
        "threepp/animation/KeyframeTrack.cpp"
        "threepp/animation/PropertyBinding.cpp"
        "threepp/animation/PropertyMixer.cpp"

        "threepp/cameras/Camera.cpp"
        "threepp/cameras/PerspectiveCamera.cpp"
        "threepp/cameras/OrthographicCamera.cpp"

        "threepp/controls/DragControls.cpp"
        "threepp/controls/FlyControls.cpp"
        "threepp/controls/OrbitControls.cpp"
        "threepp/controls/TransformControls.cpp"

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
        "threepp/helpers/DepthSensor.cpp"
        "threepp/helpers/LidarSensor.cpp"
        "threepp/helpers/DirectionalLightHelper.cpp"
        "threepp/helpers/GridHelper.cpp"
        "threepp/helpers/HemisphereLightHelper.cpp"
        "threepp/helpers/PlaneHelper.cpp"
        "threepp/helpers/PointLightHelper.cpp"
        "threepp/helpers/PolarGridHelper.cpp"
        "threepp/helpers/SkeletonHelper.cpp"
        "threepp/helpers/SpotLightHelper.cpp"

        "threepp/input/PeripheralsEventSource.cpp"

        "threepp/loaders/AsyncGroup.cpp"
        "threepp/loaders/ColladaLoader.cpp"
        "threepp/loaders/DDSLoader.cpp"
        "threepp/loaders/FontLoader.cpp"
        "threepp/loaders/GLTFLoader.cpp"
        "threepp/loaders/MaterialVariants.cpp"
        "threepp/loaders/ImageLoader.cpp"
        "threepp/loaders/RGBELoader.cpp"
        "threepp/loaders/MTLLoader.cpp"
        "threepp/loaders/ModelLoader.cpp"
        "threepp/loaders/OBJLoader.cpp"
        "threepp/loaders/STLLoader.cpp"
        "threepp/loaders/TextureLoader.cpp"
        "threepp/loaders/URDFLoader.cpp"

        "threepp/materials/LineBasicMaterial.cpp"
        "threepp/materials/LineDashedMaterial.cpp"
        "threepp/materials/Material.cpp"
        "threepp/materials/MeshBasicMaterial.cpp"
        "threepp/materials/MeshLambertMaterial.cpp"
        "threepp/materials/MeshNormalMaterial.cpp"
        "threepp/materials/MeshPhongMaterial.cpp"
        "threepp/materials/MeshPhysicalMaterial.cpp"
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
        "threepp/math/Interpolant.cpp"
        "threepp/math/Line3.cpp"
        "threepp/math/Lut.cpp"
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
        "threepp/lights/RectAreaLight.cpp"
        "threepp/lights/RectAreaLightUniformsLib.cpp"
        "threepp/lights/ltc/ltc_data.cpp"
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
        "threepp/objects/TextSprite.cpp"
        "threepp/objects/Water.cpp"
        "threepp/objects/DisplacedMesh.cpp"

        "threepp/textures/Texture.cpp"
        "threepp/textures/DataTexture3D.cpp"
        "threepp/textures/DepthTexture.cpp"

        "threepp/utils/BufferGeometryUtils.cpp"
        "threepp/utils/ImageUtils.cpp"
        "threepp/utils/StringUtils.cpp"
        "threepp/utils/TaskManager.cpp"

        "threepp/renderers/GLRenderer.cpp"
        "threepp/renderers/RenderTarget.cpp"
        "threepp/renderers/RendererFactory.cpp"

        "threepp/renderers/common/Lights.cpp"
        "threepp/renderers/common/RenderLists.cpp"
        "threepp/renderers/common/RenderStates.cpp"

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
        "threepp/renderers/gl/GLPMREM.cpp"
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
    )
endif ()

if (THREEPP_WITH_USD)
    list(APPEND publicHeaders
            "threepp/loaders/USDLoader.hpp"
    )

    list(APPEND sources
            "threepp/loaders/USDLoader.cpp"
    )
endif ()

if (THREEPP_WITH_FBX)
    list(APPEND publicHeaders
            "threepp/loaders/FBXLoader.hpp"
    )

    list(APPEND sources
            "threepp/loaders/FBXLoader.cpp"
            "${openfbx_SOURCE_DIR}/src/ofbx.cpp"
            "${openfbx_SOURCE_DIR}/src/libdeflate.c"
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

        # https://github.com/microsoft/vcpkg/blob/master/ports/glfw3/portfile.cmake
        if (UNIX AND NOT APPLE)
            message("GLFW3 currently requires the following libraries from the system package manager:
    xinerama
    xcursor
    xorg
    libglu1-mesa
    pkg-config

These can be installed on Ubuntu systems via sudo apt install libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
"
            )
        endif ()

        # https://github.com/raysan5/raylib/blob/master/cmake/GlfwImport.cmake
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
        if (BUILD_SHARED_LIBS)
            set(GLFW_LIBRARY_TYPE "SHARED" CACHE STRING "" FORCE)
        else()
            set(GLFW_LIBRARY_TYPE "OBJECT" CACHE STRING "" FORCE)
        endif()
        add_subdirectory("external/glfw")

        # Hide glfw's symbols when building a shared lib
        if (BUILD_SHARED_LIBS)
            set_property(TARGET glfw PROPERTY C_VISIBILITY_PRESET hidden)
        else ()
            list(APPEND sources $<TARGET_OBJECTS:glfw>)
        endif ()


    endif ()
endif ()

if (THREEPP_WITH_VULKAN)
    list(APPEND publicHeaders
            "threepp/renderers/VulkanRenderer.hpp"
    )
    list(APPEND privateHeaders
            "threepp/renderers/vulkan/VulkanContext.hpp"
            "threepp/renderers/vulkan/VulkanResources.hpp"
            "threepp/renderers/vulkan/water/OceanFFT.hpp"
    )
    list(APPEND sources
            "threepp/renderers/VulkanRenderer.cpp"
            "threepp/renderers/vulkan/VulkanContext.cpp"
            "threepp/renderers/vulkan/VulkanResources.cpp"
            "threepp/renderers/vulkan/water/OceanFFT.cpp"
    )
endif ()

# BCn / DXT software decoder is shared between the WGPU and Vulkan path
# tracers; compile it whenever either backend is enabled.
if (THREEPP_WITH_VULKAN OR THREEPP_WITH_WGPU)
    list(APPEND sources
            "threepp/renderers/wgpu/pathtracer/WgpuPathTracerBCn.cpp"
    )
endif ()

if (THREEPP_WITH_WGPU)
    list(APPEND publicHeaders
            "threepp/renderers/wgpu/WgpuBuffer.hpp"
            "threepp/renderers/wgpu/WgpuTexture.hpp"
            "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
            "threepp/renderers/wgpu/EffectComposer.hpp"
            "threepp/renderers/wgpu/ShaderPass.hpp"
            "threepp/renderers/wgpu/WgpuPathTracer.hpp"
    )
    list(APPEND sources
            "threepp/renderers/WgpuRenderer.cpp"
            "threepp/renderers/CrossRenderer.cpp"
            "threepp/renderers/wgpu/WgpuBindGroupCache.cpp"
            "threepp/renderers/wgpu/WgpuBindGroups.cpp"
            "threepp/renderers/wgpu/WgpuBufferPool.cpp"
            "threepp/renderers/wgpu/WgpuGeometries.cpp"
            "threepp/renderers/wgpu/WgpuLights.cpp"
            "threepp/renderers/wgpu/WgpuMaterials.cpp"
            "threepp/renderers/wgpu/WgpuPipelines.cpp"
            "threepp/renderers/wgpu/WgpuReadback.cpp"
            "threepp/renderers/wgpu/WgpuRenderTargets.cpp"
            "threepp/renderers/wgpu/WgpuShaders.cpp"
            "threepp/renderers/wgpu/WgpuShadowMap.cpp"
            "threepp/renderers/wgpu/WgpuMipmapGenerator.cpp"
            "threepp/renderers/wgpu/WgpuPMREM.cpp"
            "threepp/renderers/wgpu/WgpuTextures.cpp"
            "threepp/renderers/wgpu/WgpuTexture.cpp"
            "threepp/renderers/wgpu/WgpuBuffer.cpp"
            "threepp/renderers/wgpu/WgpuComputePipeline.cpp"
            "threepp/renderers/wgpu/EffectComposer.cpp"
            "threepp/renderers/wgpu/ShaderPass.cpp"

            "threepp/renderers/wgpu/pathtracer/WgpuPathTracer.cpp"
            "threepp/renderers/wgpu/pathtracer/WgpuPathTracerAtlas.cpp"
            "threepp/renderers/wgpu/pathtracer/WgpuPathTracerGeometry.cpp"
            "threepp/renderers/wgpu/pathtracer/WgpuPathTracerBvh.cpp"
            "threepp/renderers/wgpu/pathtracer/WgpuPathTracerShaders.cpp"
    )
    if (APPLE)
        list(APPEND sources
                "threepp/renderers/WgpuSurfaceMacOS.mm"
        )
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
        "${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/quickhull"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/stb"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/pugixml"
)

if (THREEPP_WITH_AUDIO)
    target_include_directories(threepp
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/external/miniaudio"
    )
endif ()

if (THREEPP_WITH_USD)
    message("Using tinyusdz: ${tinyusdz_SOURCE_DIR}")
    target_link_libraries(threepp PRIVATE tinyusdz_static)
    target_include_directories(threepp PRIVATE
            "${tinyusdz_SOURCE_DIR}/src"
    )
    target_compile_definitions(threepp PRIVATE THREEPP_WITH_USD)
endif ()

if (THREEPP_WITH_FBX)
    target_include_directories(threepp PRIVATE "${openfbx_SOURCE_DIR}/src")
    target_compile_definitions(threepp PRIVATE THREEPP_WITH_FBX)
endif ()

if (NOT DEFINED EMSCRIPTEN)

    target_include_directories(threepp
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/external/glad"
    )

    if (THREEPP_USE_EXTERNAL_GLFW OR BUILD_SHARED_LIBS)
        target_link_libraries(threepp PRIVATE glfw)
    else ()
        target_include_directories(threepp PRIVATE BEFORE SYSTEM "external/glfw/include")
        if (APPLE)
            target_link_libraries(threepp PRIVATE "-framework IOKit")
            target_link_libraries(threepp PRIVATE "-framework Cocoa")
            target_link_libraries(threepp PRIVATE "-framework OpenGL")
        endif ()
    endif ()

endif ()


if (THREEPP_WITH_WGPU)
    target_compile_definitions(threepp PRIVATE THREEPP_WITH_WGPU=1)

    # Embed the path tracer's .wgsl shader sources as chunked C++ raw string
    # literals at build time. Source files live under
    #   src/threepp/renderers/wgpu/pathtracer/shaders/
    # and the generated .cpp is placed under ${CMAKE_BINARY_DIR}/generated/.
    include(${PROJECT_SOURCE_DIR}/cmake/EmbedWgslPathTracer.cmake)
    embed_wgsl_path_tracer(threepp)

    if (DEFINED EMSCRIPTEN)
        # Emscripten exposes the browser's native WebGPU — no wgpu-native needed.
        target_compile_options(threepp PUBLIC "--use-port=emdawnwebgpu")

        # Pre-translate GLSL shaders to WGSL via naga for browser deployment.
        # Browsers only accept WGSL (WGPUSType_ShaderSourceSPIRV is not available).
        include(${PROJECT_SOURCE_DIR}/cmake/FetchNaga.cmake)
        include(${PROJECT_SOURCE_DIR}/cmake/EmbedWgslFromGlsl.cmake)

        embed_wgsl_from_glsl(threepp water
            "${PROJECT_SOURCE_DIR}/data/shaders/wgsl_input/water.vert"
            "${PROJECT_SOURCE_DIR}/data/shaders/wgsl_input/water.frag"
        )
    else ()
        include(${PROJECT_SOURCE_DIR}/cmake/FetchWgpuNative.cmake)

        target_include_directories(threepp PRIVATE "${WGPU_INCLUDE_DIR}")
        target_link_libraries(threepp PUBLIC ${WGPU_LIBRARY})

        if (THREEPP_WGPU_GLSL_COMPAT)
            message(STATUS "Wgpu GLSL compat: fetching glslang...")
            include(${PROJECT_SOURCE_DIR}/cmake/FetchGlslang.cmake)

            target_sources(threepp PRIVATE
                "threepp/renderers/wgpu/WgpuShaderTranslator.cpp"
            )
            target_include_directories(threepp PRIVATE "${glslang_SOURCE_DIR}")
            target_link_libraries(threepp PRIVATE
                glslang
                SPIRV
                glslang-default-resource-limits
            )
            target_compile_definitions(threepp PRIVATE THREEPP_WGPU_GLSL_COMPAT=1)
        endif()

        if (APPLE)
            target_link_libraries(threepp PRIVATE "-framework QuartzCore" "-framework Metal")
        endif ()

        # Copy wgpu_native.dll next to executables so they can find it at runtime.
        # Multi-config generators (Visual Studio) put EXEs in CMAKE_RUNTIME_OUTPUT_DIRECTORY/$<CONFIG>/,
        # while single-config generators put them directly in CMAKE_RUNTIME_OUTPUT_DIRECTORY.
        if (WIN32 AND WGPU_NATIVE_DLL)
            if (CMAKE_CONFIGURATION_TYPES)
                set(_wgpu_dll_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
            else ()
                set(_wgpu_dll_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
            endif ()
            add_custom_command(TARGET threepp POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${WGPU_NATIVE_DLL}"
                    "${_wgpu_dll_dest}/wgpu_native.dll"
                COMMENT "Copying wgpu_native.dll to output directory"
            )
        endif ()
    endif ()
endif ()


if (THREEPP_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
    target_compile_definitions(threepp PRIVATE THREEPP_WITH_VULKAN=1)
    target_link_libraries(threepp PUBLIC Vulkan::Vulkan)
    message(STATUS "Vulkan SDK: ${Vulkan_VERSION} (${Vulkan_LIBRARY})")

    # VMA single-header. SDK installs put it under <SDK>/Include/vma/, while
    # vcpkg's vulkan-memory-allocator port installs it directly at the
    # include root. Add the SDK's vma/ subdir if it exists so the source can
    # use the bare include "vk_mem_alloc.h" path either way.
    if (DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/Include/vma/vk_mem_alloc.h")
        target_include_directories(threepp PRIVATE "$ENV{VULKAN_SDK}/Include/vma")
        message(STATUS "VMA (SDK): $ENV{VULKAN_SDK}/Include/vma")
    elseif (Vulkan_INCLUDE_DIR AND EXISTS "${Vulkan_INCLUDE_DIR}/vma/vk_mem_alloc.h")
        target_include_directories(threepp PRIVATE "${Vulkan_INCLUDE_DIR}/vma")
        message(STATUS "VMA (Vulkan_INCLUDE_DIR): ${Vulkan_INCLUDE_DIR}/vma")
    else ()
        # Fall back to expecting <vk_mem_alloc.h> on the default include path
        # (vcpkg's vulkan-memory-allocator). find_path verifies it.
        find_path(VMA_INCLUDE_DIR vk_mem_alloc.h)
        if (NOT VMA_INCLUDE_DIR)
            message(FATAL_ERROR
                "Vulkan Memory Allocator (vk_mem_alloc.h) not found. "
                "Install Vulkan SDK with the VMA component, or "
                "`vcpkg install vulkan-memory-allocator`.")
        endif ()
        target_include_directories(threepp PRIVATE "${VMA_INCLUDE_DIR}")
        message(STATUS "VMA: ${VMA_INCLUDE_DIR}")
    endif ()

    include(${PROJECT_SOURCE_DIR}/cmake/CompileVulkanShaders.cmake)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/raygen.rgen"
        kRaygenRgenSpv
        _rgen_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/miss.rmiss"
        kMissRmissSpv
        _miss_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/shadow_miss.rmiss"
        kShadowMissRmissSpv
        _shadow_miss_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/closest_hit.rchit"
        kClosestHitRchitSpv
        _chit_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/closest_hit_alpha.rahit"
        kClosestHitAlphaRahitSpv
        _ahit_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/shadow_anyhit.rahit"
        kShadowAnyhitRahitSpv
        _shadow_ahit_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/photon_emit.rgen"
        kPhotonEmitRgenSpv
        _photon_emit_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/photon_miss.rmiss"
        kPhotonMissRmissSpv
        _photon_miss_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/photon_chit.rchit"
        kPhotonChitRchitSpv
        _photon_chit_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/prefilter_env.comp"
        kPrefilterEnvCompSpv
        _prefilter_env_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/denoise.comp"
        kDenoiseCompSpv
        _denoise_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/denoise_atrous.comp"
        kDenoiseAtrousCompSpv
        _denoise_atrous_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/skinning.comp"
        kSkinningCompSpv
        _skinning_spv_h)
    # FFT-based ocean (DisplacedMesh): Phillips spectrum, time-evolution,
    # IFFT (twiddle table + horizontal/vertical butterflies + permute) and
    # a vertex-displacement compute that writes into the BLAS vertex buffer.
    # Each shader is independent of the path tracer and only consumed by the
    # OceanFFT host classes (src/threepp/renderers/vulkan/water/).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/phillips_spectrum.comp"
        kPhillipsSpectrumCompSpv
        _phillips_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/dynamic_spectrum.comp"
        kDynamicSpectrumCompSpv
        _dynamic_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/ifft_twiddle.comp"
        kIfftTwiddleCompSpv
        _ifft_twiddle_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/ifft_horizontal.comp"
        kIfftHorizontalCompSpv
        _ifft_horizontal_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/ifft_vertical.comp"
        kIfftVerticalCompSpv
        _ifft_vertical_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/ifft_permute.comp"
        kIfftPermuteCompSpv
        _ifft_permute_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/water_displace.comp"
        kWaterDisplaceCompSpv
        _water_displace_spv_h)
    # Hybrid raster G-buffer prepass — produces depth/normal/motion/IDs
    # for the path tracer to skip its primary BVH trace. See raygen.rgen
    # for the consumer side.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/gbuffer.vert"
        kGbufferVertSpv
        _gbuffer_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/gbuffer.frag"
        kGbufferFragSpv
        _gbuffer_frag_spv_h)
    # Indirect-drawing variant of the G-buffer vertex shader. Reads per-mesh
    # data (model matrix, attribute buffer device addresses, flags) from an
    # SSBO indexed by gl_DrawIDEXT, fetches vertices via buffer references —
    # lets the whole gbuf pass ship as 1-3 vkCmdDrawIndirect calls instead
    # of N vkCmdDrawIndexed calls. Path B of the perf work; see
    # recordRasterGbufPass for the dispatch side.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/gbuffer_indirect.vert"
        kGbufferIndirectVertSpv
        _gbuffer_indirect_vert_spv_h)
    # Hybrid raster TAA: post-denoise temporal resolve. Reads taaInput +
    # history + gbufMotion → writes swapchain + new history. Runs only
    # when hybridEnabled_; non-hybrid path copies taaInput → swapchain.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/taa_resolve.comp"
        kTaaResolveCompSpv
        _taa_resolve_spv_h)
    # Hybrid raster overlay: wireframe + Line/LineSegments draw on top of
    # the post-TAA swapchain image. Depth-tested against the G-buffer's
    # depth attachment so overlays are correctly occluded by path-traced
    # geometry. Only runs when hybridEnabled_ + scene has overlay objects.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay.vert"
        kOverlayVertSpv
        _overlay_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay.frag"
        kOverlayFragSpv
        _overlay_frag_spv_h)
    # Overlay depth prepass — writes unjittered depth so the wireframe
    # overlay's depth test compares like-with-like (the main G-buffer
    # depth is jittered for TAA).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_depth.vert"
        kOverlayDepthVertSpv
        _overlay_depth_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_depth.frag"
        kOverlayDepthFragSpv
        _overlay_depth_frag_spv_h)
    # Overlay variant with per-vertex color input — used by Line /
    # LineSegments when geometry has a "color" attribute AND material
    # has vertexColors=true (e.g. AxesHelper).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_color.vert"
        kOverlayColorVertSpv
        _overlay_color_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_color.frag"
        kOverlayColorFragSpv
        _overlay_color_frag_spv_h)
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
