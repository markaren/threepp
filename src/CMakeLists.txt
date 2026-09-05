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
        "threepp/controls/TrackballControls.hpp"
        "threepp/controls/TransformControls.hpp"

        "threepp/core/Assert.hpp"
        "threepp/core/AttributeView.hpp"
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
        "threepp/extras/conveyor/ConveyorGeometry.hpp"
        "threepp/extras/effects/FireEffect.hpp"
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
        "threepp/helpers/CameraSensor.hpp"
        "threepp/helpers/DepthSensor.hpp"
        "threepp/helpers/EventCameraSensor.hpp"
        "threepp/helpers/LidarModel.hpp"
        "threepp/helpers/SonarModel.hpp"
        "threepp/helpers/LidarSensor.hpp"
        "threepp/helpers/LidarTypes.hpp"
        "threepp/helpers/LidarWaveform.hpp"
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
        "threepp/loaders/EXRLoader.hpp"
        "threepp/loaders/FontLoader.hpp"
        "threepp/loaders/GLTFLoader.hpp"
        "threepp/loaders/MaterialVariants.hpp"
        "threepp/loaders/MTLLoader.hpp"
        "threepp/loaders/ImageLoader.hpp"
        "threepp/loaders/RGBELoader.hpp"
        "threepp/loaders/ModelLoader.hpp"
        "threepp/loaders/ObjectExporter.hpp"
        "threepp/loaders/ObjectLoader.hpp"
        "threepp/loaders/OBJLoader.hpp"
        "threepp/loaders/STLLoader.hpp"
        "threepp/loaders/SogLoader.hpp"
        "threepp/loaders/SplatLoader.hpp"
        "threepp/loaders/SVGLoader.hpp"
        "threepp/loaders/TextureLoader.hpp"
        "threepp/loaders/URDFLoader.hpp"
        "threepp/loaders/Xacro.hpp"

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
        "threepp/objects/GrassMesh.hpp"
        "threepp/objects/ParticleField.hpp"
        "threepp/objects/SplatCloud.hpp"

        "threepp/splats/PointSurface.hpp"
        "threepp/splats/SplatData.hpp"
        "threepp/splats/SplatLod.hpp"
        "threepp/splats/SplatSH.hpp"

        "threepp/textures/CubeTexture.hpp"
        "threepp/textures/DataTexture.hpp"
        "threepp/textures/DataTexture3D.hpp"
        "threepp/textures/DepthTexture.hpp"
        "threepp/textures/FramebufferTexture.hpp"
        "threepp/textures/Image.hpp"
        "threepp/textures/Texture.hpp"

        "threepp/utils/BufferGeometryUtils.hpp"
        "threepp/utils/BVH.hpp"
        "threepp/utils/GeometryLod.hpp"
        "threepp/utils/ImageUtils.hpp"
        "threepp/utils/StringUtils.hpp"
        "threepp/utils/TaskManager.hpp"
        "threepp/utils/ZipReader.hpp"
        "threepp/utils/ZipWriter.hpp"

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
        "threepp/renderers/GLRenderer.hpp"
        "threepp/renderers/RenderTarget.hpp"
        "threepp/renderers/GLRenderTarget.hpp"

        "threepp/renderers/gl/GLInfo.hpp"
        "threepp/renderers/gl/GLShadowMap.hpp"
        "threepp/renderers/gl/GLState.hpp"

        "threepp/renderers/shaders/ShaderChunk.hpp"
        "threepp/renderers/shaders/ShaderLib.hpp"
        "threepp/renderers/shaders/UniformsLib.hpp"

        "threepp/postprocessing/postprocessing.hpp"
        "threepp/postprocessing/BokehPass.hpp"
        "threepp/postprocessing/ClearPass.hpp"
        "threepp/postprocessing/EffectComposer.hpp"
        "threepp/postprocessing/MaskPass.hpp"
        "threepp/postprocessing/Pass.hpp"
        "threepp/postprocessing/RenderPass.hpp"
        "threepp/postprocessing/SavePass.hpp"
        "threepp/postprocessing/ShaderPass.hpp"
        "threepp/postprocessing/TexturePass.hpp"
        "threepp/postprocessing/UnrealBloomPass.hpp"
        "threepp/postprocessing/shaders/BokehShader.hpp"
        "threepp/postprocessing/shaders/CopyShader.hpp"
        "threepp/postprocessing/shaders/LuminosityHighPassShader.hpp"

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

        "threepp/utils/Base64.hpp"
        "threepp/utils/RegexUtil.hpp"
        "threepp/utils/TriangleIntersect.hpp"

        "threepp/loaders/HdrTexture.hpp"
        "threepp/loaders/ObjectJsonConstants.hpp"

        "threepp/loaders/exr/PizDecode.hpp"

        "threepp/loaders/xacro/Diagnostics.hpp"
        "threepp/loaders/xacro/Expand.hpp"
        "threepp/loaders/xacro/Expr.hpp"
        "threepp/loaders/xacro/PackageResolver.hpp"
        "threepp/loaders/xacro/Scope.hpp"
        "threepp/loaders/xacro/Substitution.hpp"
        "threepp/loaders/xacro/Value.hpp"
        "threepp/loaders/xacro/YamlLite.hpp"
)

# libwebp v1.6.0, decoder subset — exactly upstream's own `webpdecoder` target, compiled
# straight into threepp the way meshoptimizer and pugixml are. Upstream's CMake is not used:
# add_subdirectory would bring install rules, pkg-config generation and package exports with
# it. Provenance and the manifest these names come from: external/libwebp/README.threepp.md.
set(libwebpSources

        "external/libwebp/src/dec/alpha_dec.c"
        "external/libwebp/src/dec/buffer_dec.c"
        "external/libwebp/src/dec/frame_dec.c"
        "external/libwebp/src/dec/idec_dec.c"
        "external/libwebp/src/dec/io_dec.c"
        "external/libwebp/src/dec/quant_dec.c"
        "external/libwebp/src/dec/tree_dec.c"
        "external/libwebp/src/dec/vp8_dec.c"
        "external/libwebp/src/dec/vp8l_dec.c"
        "external/libwebp/src/dec/webp_dec.c"

        # The SIMD kernels are all listed unconditionally: each one is wrapped in its own
        # WEBP_USE_<arch> guard and compiles to nothing off-target, and libwebp picks between
        # them at runtime. No per-file -msse4.1/-mavx2 here — without those flags the SSE4.1
        # and AVX2 kernels self-disable and dispatch falls back, which is the intended
        # degradation rather than a build error.
        "external/libwebp/src/dsp/alpha_processing.c"
        "external/libwebp/src/dsp/alpha_processing_mips_dsp_r2.c"
        "external/libwebp/src/dsp/alpha_processing_neon.c"
        "external/libwebp/src/dsp/alpha_processing_sse2.c"
        "external/libwebp/src/dsp/alpha_processing_sse41.c"
        "external/libwebp/src/dsp/cpu.c"
        "external/libwebp/src/dsp/dec.c"
        "external/libwebp/src/dsp/dec_clip_tables.c"
        "external/libwebp/src/dsp/dec_mips32.c"
        "external/libwebp/src/dsp/dec_mips_dsp_r2.c"
        "external/libwebp/src/dsp/dec_msa.c"
        "external/libwebp/src/dsp/dec_neon.c"
        "external/libwebp/src/dsp/dec_sse2.c"
        "external/libwebp/src/dsp/dec_sse41.c"
        "external/libwebp/src/dsp/filters.c"
        "external/libwebp/src/dsp/filters_mips_dsp_r2.c"
        "external/libwebp/src/dsp/filters_msa.c"
        "external/libwebp/src/dsp/filters_neon.c"
        "external/libwebp/src/dsp/filters_sse2.c"
        "external/libwebp/src/dsp/lossless.c"
        "external/libwebp/src/dsp/lossless_avx2.c"
        "external/libwebp/src/dsp/lossless_mips_dsp_r2.c"
        "external/libwebp/src/dsp/lossless_msa.c"
        "external/libwebp/src/dsp/lossless_neon.c"
        "external/libwebp/src/dsp/lossless_sse2.c"
        "external/libwebp/src/dsp/lossless_sse41.c"
        "external/libwebp/src/dsp/rescaler.c"
        "external/libwebp/src/dsp/rescaler_mips32.c"
        "external/libwebp/src/dsp/rescaler_mips_dsp_r2.c"
        "external/libwebp/src/dsp/rescaler_msa.c"
        "external/libwebp/src/dsp/rescaler_neon.c"
        "external/libwebp/src/dsp/rescaler_sse2.c"
        "external/libwebp/src/dsp/upsampling.c"
        "external/libwebp/src/dsp/upsampling_mips_dsp_r2.c"
        "external/libwebp/src/dsp/upsampling_msa.c"
        "external/libwebp/src/dsp/upsampling_neon.c"
        "external/libwebp/src/dsp/upsampling_sse2.c"
        "external/libwebp/src/dsp/upsampling_sse41.c"
        "external/libwebp/src/dsp/yuv.c"
        "external/libwebp/src/dsp/yuv_mips32.c"
        "external/libwebp/src/dsp/yuv_mips_dsp_r2.c"
        "external/libwebp/src/dsp/yuv_neon.c"
        "external/libwebp/src/dsp/yuv_sse2.c"
        "external/libwebp/src/dsp/yuv_sse41.c"

        "external/libwebp/src/utils/bit_reader_utils.c"
        "external/libwebp/src/utils/color_cache_utils.c"
        "external/libwebp/src/utils/filters_utils.c"
        "external/libwebp/src/utils/huffman_utils.c"
        "external/libwebp/src/utils/palette.c"
        "external/libwebp/src/utils/quant_levels_dec_utils.c"
        "external/libwebp/src/utils/random_utils.c"
        "external/libwebp/src/utils/rescaler_utils.c"
        "external/libwebp/src/utils/thread_utils.c"
        "external/libwebp/src/utils/utils.c"
)

# Upstream's C is not ours to keep warning-clean, and THREEPP_TREAT_WARNINGS_AS_ERRORS would
# otherwise let one of its diagnostics fail the whole build.
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set_source_files_properties(${libwebpSources} PROPERTIES COMPILE_OPTIONS "-w")
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(${libwebpSources} PROPERTIES COMPILE_OPTIONS "/w")
endif ()

set(sources

        ${libwebpSources}

        "external/pugixml/pugixml.cpp"
        "external/meshoptimizer/allocator.cpp"
        "external/meshoptimizer/indexcodec.cpp"
        "external/meshoptimizer/indexgenerator.cpp"
        "external/meshoptimizer/simplifier.cpp"
        "external/meshoptimizer/vertexcodec.cpp"
        "external/meshoptimizer/vertexfilter.cpp"

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
        "threepp/controls/TrackballControls.cpp"
        "threepp/controls/TransformControls.cpp"

        "threepp/core/AttributeView.cpp"
        "threepp/core/BufferGeometry.cpp"
        "threepp/core/Clock.cpp"
        "threepp/core/EventDispatcher.cpp"
        "threepp/core/Layers.cpp"
        "threepp/core/Object3D.cpp"
        "threepp/core/Raycaster.cpp"
        "threepp/core/Uniform.cpp"

        "threepp/extras/ShapeUtils.cpp"
        "threepp/extras/terrain/TerrainGenerator.cpp"
        "threepp/extras/terrain/GeoTerrainPack.cpp"
        "threepp/extras/uav/DownwashEffect.cpp"
        "threepp/extras/uav/MavlinkOut.cpp"
        "threepp/extras/uav/SitlBridge.cpp"
        "threepp/extras/core/Curve.cpp"
        "threepp/extras/core/CurvePath.cpp"
        "threepp/extras/core/Font.cpp"
        "threepp/extras/core/Path.cpp"
        "threepp/extras/core/Shape.cpp"
        "threepp/extras/core/ShapePath.cpp"
        "threepp/extras/conveyor/ConveyorGeometry.cpp"
        "threepp/extras/effects/FireEffect.cpp"
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
        "threepp/helpers/CameraSensor.cpp"
        "threepp/helpers/DepthSensor.cpp"
        "threepp/helpers/EventCameraSensor.cpp"
        "threepp/helpers/LidarSensor.cpp"
        "threepp/helpers/SonarModel.cpp"
        "threepp/helpers/LidarWaveform.cpp"
        "threepp/helpers/TracedRasterVisionSensor.cpp"
        "threepp/helpers/DirectionalLightHelper.cpp"
        "threepp/helpers/GridHelper.cpp"
        "threepp/helpers/HemisphereLightHelper.cpp"
        "threepp/helpers/PlaneHelper.cpp"
        "threepp/helpers/PointLightHelper.cpp"
        "threepp/helpers/PolarGridHelper.cpp"
        "threepp/helpers/SkeletonHelper.cpp"
        "threepp/helpers/SpotLightHelper.cpp"

        "threepp/input/PeripheralsEventSource.cpp"

        "threepp/loaders/AssetSource.cpp"
        "threepp/loaders/AsyncGroup.cpp"
        "threepp/loaders/ColladaLoader.cpp"
        "threepp/loaders/DDSLoader.cpp"
        "threepp/loaders/EXRLoader.cpp"
        "threepp/loaders/exr/PizDecode.cpp"
        "threepp/loaders/FontLoader.cpp"
        "threepp/loaders/GLTFLoader.cpp"
        "threepp/loaders/MaterialVariants.cpp"
        "threepp/loaders/ImageLoader.cpp"
        "threepp/loaders/RGBELoader.cpp"
        "threepp/loaders/MTLLoader.cpp"
        "threepp/loaders/ModelLoader.cpp"
        "threepp/loaders/ObjectExporter.cpp"
        "threepp/loaders/ObjectLoader.cpp"
        "threepp/loaders/OBJLoader.cpp"
        "threepp/loaders/STLLoader.cpp"
        "threepp/loaders/SogLoader.cpp"
        "threepp/loaders/SplatLoader.cpp"
        "threepp/loaders/SVGLoader.cpp"
        "threepp/loaders/TextureLoader.cpp"
        "threepp/loaders/URDFLoader.cpp"

        "threepp/loaders/xacro/Diagnostics.cpp"
        "threepp/loaders/xacro/Expand.cpp"
        "threepp/loaders/xacro/Expr.cpp"
        "threepp/loaders/xacro/PackageResolver.cpp"
        "threepp/loaders/xacro/Scope.cpp"
        "threepp/loaders/xacro/Substitution.cpp"
        "threepp/loaders/xacro/Value.cpp"
        "threepp/loaders/xacro/Xacro.cpp"
        "threepp/loaders/xacro/YamlLite.cpp"

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
        "threepp/objects/GrassMesh.cpp"
        "threepp/objects/ParticleField.cpp"
        "threepp/objects/SplatCloud.cpp"

        "threepp/splats/PointSurface.cpp"
        "threepp/splats/SplatData.cpp"
        "threepp/splats/SplatLod.cpp"

        "threepp/textures/Texture.cpp"
        "threepp/textures/DataTexture3D.cpp"
        "threepp/textures/DepthTexture.cpp"

        "threepp/utils/Base64.cpp"
        "threepp/utils/BufferGeometryUtils.cpp"
        "threepp/utils/BVH.cpp"
        "threepp/utils/GeometryLod.cpp"
        "threepp/utils/ImageUtils.cpp"
        "threepp/utils/StbImageWrite.cpp"
        "threepp/utils/StringUtils.cpp"
        "threepp/utils/TaskManager.cpp"
        "threepp/utils/TriangleIntersect.cpp"
        "threepp/utils/ZipReader.cpp"
        "threepp/utils/ZipWriter.cpp"

        "threepp/renderers/GLRenderer.cpp"
        "threepp/renderers/RenderTarget.cpp"

        "threepp/postprocessing/BokehPass.cpp"
        "threepp/postprocessing/ClearPass.cpp"
        "threepp/postprocessing/EffectComposer.cpp"
        "threepp/postprocessing/MaskPass.cpp"
        "threepp/postprocessing/Pass.cpp"
        "threepp/postprocessing/RenderPass.cpp"
        "threepp/postprocessing/SavePass.cpp"
        "threepp/postprocessing/ShaderPass.cpp"
        "threepp/postprocessing/TexturePass.cpp"
        "threepp/postprocessing/UnrealBloomPass.cpp"

        "threepp/renderers/common/Lights.cpp"
        "threepp/renderers/common/RenderLists.cpp"
        "threepp/renderers/common/RenderStates.cpp"
        # Pure-CPU BCn encode/decode. Only the Vulkan renderer consumes these,
        # but they have no Vulkan dependency and BC7Encode_test round-trips
        # them unconditionally — keeping them in every build lets CI without a
        # Vulkan SDK still pin the encoder/decoder conventions.
        "threepp/renderers/common/BC7Encode.cpp"
        "threepp/renderers/common/BCnDecode.cpp"

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
            "threepp/audio/Acoustics.hpp"
            "threepp/audio/Audio.hpp"
    )

    list(APPEND sources
            "threepp/audio/Acoustics.cpp"
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
            "threepp/renderers/VulkanRendererCore.hpp"
            "threepp/renderers/VulkanRenderer.hpp"
            # Validation-layer counters. Public, and free of every Vulkan header,
            # so a test (or a consumer) can assert spec-cleanliness without the
            # PRIVATE Vulkan/VMA include paths this target keeps to itself.
            "threepp/renderers/vulkan/ValidationReport.hpp"
            "threepp/helpers/PathTracedLidarSensor.hpp"
            "threepp/helpers/SonarSensor.hpp"
            # Depth fusion of a splat scan into a triangle surface: it reads the
            # Vulkan depth AOV, so it lives with the renderer that exports it.
            "threepp/splats/SplatSurface.hpp"
            # FFT-displaced ocean: the renderer is the only consumer (recognised
            # via dynamic_cast), so the object types live with it under Vulkan.
            "threepp/objects/DisplacedMesh.hpp"
            "threepp/objects/Ocean.hpp"
    )
    list(APPEND privateHeaders
            "threepp/renderers/vulkan/VulkanContext.hpp"
            "threepp/renderers/vulkan/VulkanResources.hpp"
            "threepp/renderers/vulkan/AutoExposure.hpp"
            "threepp/renderers/vulkan/GbufResolve.hpp"
            "threepp/renderers/vulkan/LidarScanner.hpp"
            "threepp/renderers/vulkan/EnvPrefilter.hpp"
            "threepp/renderers/vulkan/EventCameraDetector.hpp"
            "threepp/renderers/vulkan/SkinningPipeline.hpp"
            "threepp/renderers/vulkan/TetSkinningPipeline.hpp"
            "threepp/renderers/vulkan/GpuTimings.hpp"
            "threepp/renderers/vulkan/OverlayPass.hpp"
            "threepp/renderers/vulkan/VulkanFrameTypes.hpp"
            # Types split out of the VulkanCoreImpl.hpp Impl header.
            "threepp/renderers/vulkan/VulkanImplCommon.hpp"
            "threepp/renderers/vulkan/VulkanGpuLayouts.hpp"
            "threepp/renderers/vulkan/VulkanGeometryState.hpp"
            "threepp/renderers/vulkan/VulkanSceneTypes.hpp"
            "threepp/renderers/vulkan/ParticleFieldPass.hpp"
            "threepp/renderers/vulkan/VulkanViewContext.hpp"
            "threepp/renderers/vulkan/TaaResolve.hpp"
            "threepp/renderers/vulkan/WaterDisplacePipeline.hpp"
            "threepp/renderers/vulkan/GrassWindPipeline.hpp"
            "threepp/renderers/vulkan/water/OceanFFT.hpp"
    )
    list(APPEND sources
            "threepp/renderers/VulkanRenderer.cpp"
            "threepp/helpers/PathTracedLidarSensor.cpp"
            "threepp/helpers/SonarSensor.cpp"
            "threepp/splats/SplatSurface.cpp"
            "threepp/objects/DisplacedMesh.cpp"
            "threepp/objects/Ocean.cpp"
            "threepp/renderers/vulkan/VmaImpl.cpp"
            "threepp/renderers/vulkan/VulkanContext.cpp"
            "threepp/renderers/vulkan/VulkanResources.cpp"
            "threepp/renderers/vulkan/VulkanCoreInit.cpp"
            "threepp/renderers/vulkan/VulkanCoreGeometry.cpp"
            "threepp/renderers/vulkan/VulkanCoreScene.cpp"
            "threepp/renderers/vulkan/VulkanCoreUploads.cpp"
            "threepp/renderers/vulkan/VulkanCoreIndirect.cpp"
            "threepp/renderers/vulkan/VulkanCorePipelines.cpp"
            "threepp/renderers/vulkan/VulkanCoreDescriptors.cpp"
            "threepp/renderers/vulkan/VulkanCoreTextures.cpp"
            "threepp/renderers/vulkan/VulkanCoreRecord.cpp"
            "threepp/renderers/vulkan/VulkanCoreFrame.cpp"
            "threepp/renderers/vulkan/LidarScanner.cpp"
            "threepp/renderers/vulkan/EnvPrefilter.cpp"
            "threepp/renderers/vulkan/EventCameraDetector.cpp"
            "threepp/renderers/vulkan/SkinningPipeline.cpp"
            "threepp/renderers/vulkan/TetSkinningPipeline.cpp"
            "threepp/renderers/vulkan/GpuTimings.cpp"
            "threepp/renderers/vulkan/OverlayPass.cpp"
            "threepp/renderers/vulkan/TaaResolve.cpp"
            "threepp/renderers/vulkan/AutoExposure.cpp"
            "threepp/renderers/vulkan/GbufResolve.cpp"
            "threepp/renderers/vulkan/BloomPass.cpp"
            "threepp/renderers/vulkan/BillboardGlowPass.cpp"
            "threepp/renderers/vulkan/PostComposite.cpp"
            "threepp/renderers/vulkan/DofPass.cpp"
            "threepp/renderers/vulkan/SensorPass.cpp"
            "threepp/renderers/vulkan/SplatPass.cpp"
            "threepp/renderers/vulkan/DeferredShade.cpp"
            "threepp/renderers/vulkan/ProbeGI.cpp"
            "threepp/renderers/vulkan/HiZPyramid.cpp"
            "threepp/renderers/vulkan/OcclusionCull.cpp"
            "threepp/renderers/vulkan/InstanceExpand.cpp"
            "threepp/renderers/vulkan/ParticleFieldPass.cpp"
            "threepp/renderers/vulkan/WaterDisplacePipeline.cpp"
            "threepp/renderers/vulkan/FoamWorldPipeline.cpp"
            "threepp/renderers/vulkan/GrassWindPipeline.cpp"
            "threepp/renderers/vulkan/VertexSanitizePipeline.cpp"
            "threepp/renderers/vulkan/water/OceanFFT.cpp"
    )
    # AMD FidelityFX FSR 3.1 upscaler — compiled in only when enabled (it
    # includes the ffx-api headers). Replaces the built-in TAA upsampler at
    # runtime when the FSR context creates successfully; see FsrUpscaler.{hpp,cpp}.
    if (THREEPP_WITH_FSR)
        list(APPEND privateHeaders "threepp/renderers/vulkan/FsrUpscaler.hpp")
        list(APPEND sources        "threepp/renderers/vulkan/FsrUpscaler.cpp")
    endif ()
    # NVIDIA DLSS Super Resolution — compiled in only when enabled (it includes
    # the NGX headers). Replaces the built-in TAA upsampler (and outranks FSR)
    # at runtime when the NGX feature creates successfully; see DlssUpscaler.{hpp,cpp}.
    if (THREEPP_WITH_DLSS)
        list(APPEND privateHeaders "threepp/renderers/vulkan/DlssUpscaler.hpp")
        list(APPEND sources        "threepp/renderers/vulkan/DlssUpscaler.cpp")
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

# The embedded GL shader headers. Listed as target sources so their custom
# commands join the build graph — that is what makes editing a .glsl actually
# rebuild, instead of silently keeping the text embedded at configure time.
list(APPEND sources ${THREEPP_GENERATED_SHADER_HEADERS})

add_library(threepp ${sources} ${privateHeaders} ${publicHeadersFull})
add_library(threepp::threepp ALIAS threepp)
target_compile_features(threepp PUBLIC "cxx_std_20")

if (UNIX)
    target_link_libraries(threepp PRIVATE pthread dl)
endif ()

if (WIN32)
    # extras/uav/SitlBridge.cpp and MavlinkOut.cpp keep their winsock usage
    # inside the library, so consumers of the headers need neither
    # <winsock2.h> nor this link.
    target_link_libraries(threepp PRIVATE ws2_32)
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
        "${CMAKE_CURRENT_SOURCE_DIR}/external/meshoptimizer"
)

# libwebp's internal includes are rooted at its package directory — vp8l_dec.c asks for
# "src/dec/vp8li_dec.h", not "vp8li_dec.h" — so the package directory itself goes on the
# include path, and BEFORE everything else so that no other root can answer a "src/..." query
# first. BEFORE has to precede the scope keyword: `PRIVATE BEFORE` parses BEFORE as a
# directory name and appends instead of prepending.
# The nested "src" entry is for the public header, which ImageLoader.cpp includes the way an
# outside consumer would, as "webp/decode.h".
target_include_directories(threepp BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/external/libwebp"
        "${CMAKE_CURRENT_SOURCE_DIR}/external/libwebp/src"
)

if (THREEPP_WITH_AUDIO)
    target_include_directories(threepp
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/external/miniaudio"
    )
    # PUBLIC, unlike the other THREEPP_WITH_* macros: threepp/audio/Audio.hpp
    # and AudioPlaySession.hpp only EXIST in this configuration, so a consumer
    # (the editor) has to be able to ask. PUBLIC rather than per-target is also
    # what keeps the library and every consumer compiling AudioPlaySession.hpp
    # the same way — a per-target macro on a header branch is an ODR violation
    # waiting to happen (see the V-HACD note in apps/editor/CMakeLists.txt).
    target_compile_definitions(threepp PUBLIC THREEPP_WITH_AUDIO=1)
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
        target_include_directories(threepp SYSTEM BEFORE PRIVATE "external/glfw/include")
        if (APPLE)
            target_link_libraries(threepp PRIVATE "-framework IOKit")
            target_link_libraries(threepp PRIVATE "-framework Cocoa")
            target_link_libraries(threepp PRIVATE "-framework OpenGL")
        endif ()
    endif ()

endif ()


if (THREEPP_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
    # PUBLIC, because PUBLIC HEADERS branch on it: ImguiContext.hpp includes
    # imgui_impl_vulkan.h + VulkanRenderer.hpp behind it, RendererSettings.hpp
    # shows the Vulkan knobs behind it, and the header-only play sessions
    # (ParticleField, Granular, SplatSurface) drive the Vulkan backend behind it.
    # A consumer that does not see this macro therefore compiles a DIFFERENT
    # version of those headers than the library was built against — it silently
    # gets the OpenGL-only half, and two consumer targets in one binary that
    # disagree about it violate the ODR. Vulkan::Vulkan is already PUBLIC here,
    # so the include paths a consumer needs to compile those branches come along.
    target_compile_definitions(threepp PUBLIC THREEPP_WITH_VULKAN=1)
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

    # AMD FidelityFX FSR 3.1 upscaler (Windows-only, gated on THREEPP_WITH_FSR).
    # Fetches the SDK and links the prebuilt signed Vulkan ffx-api runtime; the
    # DLL ships every backend + precompiled shader, so there is no shader-build
    # step. The DLL is copied next to executables.
    if (THREEPP_WITH_FSR)
        include(${PROJECT_SOURCE_DIR}/cmake/FetchFidelityFX.cmake)
        target_include_directories(threepp PRIVATE "${FFX_INCLUDE_DIR}")
        # NB: the ffx-api entry points are resolved at runtime via LoadLibrary +
        # GetProcAddress (ffxLoadFunctions), NOT the import library — ffx_api.h
        # hard-codes __declspec(dllexport) for consumers, so linking + calling
        # directly produces a null forwarder (crash at address 0). Only the DLL
        # (copied below) is needed at runtime; the .lib is intentionally unused.
        target_compile_definitions(threepp PRIVATE THREEPP_WITH_FSR=1)

        if (WIN32 AND FFX_DLL)
            if (CMAKE_CONFIGURATION_TYPES)
                set(_ffx_dll_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
            else ()
                set(_ffx_dll_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
            endif ()
            add_custom_command(TARGET threepp POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FFX_DLL}"
                    "${_ffx_dll_dest}/amd_fidelityfx_vk.dll"
                COMMENT "Copying amd_fidelityfx_vk.dll to output directory"
            )
        endif ()
    endif ()

    # NVIDIA DLSS Super Resolution (Windows-only, gated on THREEPP_WITH_DLSS).
    # Links the NGX static import lib (proper static linkage — no FFX-style
    # dllexport trap) and copies the signed production nvngx_dlss.dll next to
    # executables; the NGX runtime LoadLibrary's it at init (DlssUpscaler passes
    # the module directory as an explicit NGX search path for the .pyd case).
    if (THREEPP_WITH_DLSS)
        include(${PROJECT_SOURCE_DIR}/cmake/FetchDLSS.cmake)
        target_include_directories(threepp PRIVATE "${NGX_INCLUDE_DIR}")
        target_compile_definitions(threepp PRIVATE THREEPP_WITH_DLSS=1)
        # _d = dynamic CRT (/MD), matching threepp's default MSVC runtime;
        # the _dbg variant pairs with /MDd in Debug configs.
        target_link_libraries(threepp PRIVATE
            "$<IF:$<CONFIG:Debug>,${NGX_LIBRARY_DBG},${NGX_LIBRARY_REL}>")

        if (WIN32 AND NGX_DLSS_DLL)
            if (CMAKE_CONFIGURATION_TYPES)
                set(_ngx_dll_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>")
            else ()
                set(_ngx_dll_dest "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
            endif ()
            add_custom_command(TARGET threepp POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${NGX_DLSS_DLL}"
                    "${_ngx_dll_dest}/nvngx_dlss.dll"
                COMMENT "Copying nvngx_dlss.dll to output directory"
            )
        endif ()
    endif ()

    include(${PROJECT_SOURCE_DIR}/cmake/CompileVulkanShaders.cmake)
    # Path-traced LIDAR scanner — private ray-tracing pipeline that emits
    # one beam per launched thread, evaluates the LIDAR equation in the
    # closest-hit shader, and writes per-beam returns back to the host
    # via a small SSBO. See vulkan/LidarScanner.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/lidar.rgen"
        kLidarRgenSpv
        _lidar_rgen_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/lidar.rchit"
        kLidarRchitSpv
        _lidar_rchit_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/lidar.rmiss"
        kLidarRmissSpv
        _lidar_rmiss_spv_h)
    # GPU event camera (DVS) detector — reads sceneCaptureBuf written by
    # the renderer's per-frame scene-capture path, updates per-pixel
    # log-luminance history, paints an RGBA8 accumulator image, copies
    # it into the host-visible readback ring.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/event_detect.comp"
        kEventDetectCompSpv
        _event_detect_spv_h)
    # Fast deterministic shading for the event-camera pipeline. Reads the
    # raster G-buffer + material/light buffers, outputs a packed BGRA8
    # luma buffer matching the swapchain layout — feeds event_detect
    # without the path tracer's stochastic noise.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/event_shade.comp"
        kEventShadeCompSpv
        _event_shade_spv_h)
    # Debug-view resolve: samples one G-buffer channel and writes a readable
    # visualization to the swapchain (setHybridDebugView). Replaces a raw blit
    # that couldn't show the signed motion or integer ids attachments.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/debug_resolve.comp"
        kDebugResolveCompSpv
        _debug_resolve_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/prefilter_env.comp"
        kPrefilterEnvCompSpv
        _prefilter_env_spv_h)
    # Post stack: progressive bloom pyramid (Jimenez 2014 — Karis 13-tap down
    # chain + 3x3 tent additive upsample, vulkan/BloomPass.{hpp,cpp}) + the
    # exposure/white-balance/tone-map/grade/sRGB composite that closes the
    # HDR path (vulkan/PostComposite.{hpp,cpp}).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/bloom_down.comp"
        kBloomDownCompSpv
        _bloom_down_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/bloom_up.comp"
        kBloomUpCompSpv
        _bloom_up_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/post_composite.comp"
        kPostCompositeCompSpv
        _post_composite_spv_h)
    # Camera image formation: lens distortion + sensor noise on the FINISHED
    # frame, after the overlay pass so it covers everything the camera sees.
    # See vulkan/SensorPass.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/sensor_image.comp"
        kSensorImageCompSpv
        _sensor_image_spv_h)
    # Thin-lens depth of field (CoC prefilter → tile max/dilate → near/far
    # scatter-as-gather → full-res combine). See vulkan/DofPass.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/dof_coc.comp"
        kDofCocCompSpv
        _dof_coc_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/dof_tile.comp"
        kDofTileCompSpv
        _dof_tile_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/dof_gather.comp"
        kDofGatherCompSpv
        _dof_gather_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/dof_combine.comp"
        kDofCombineCompSpv
        _dof_combine_spv_h)
    # Gaussian-splat compute tile rasterizer (project -> prefix sum ->
    # deterministic expansion -> radix sort -> per-tile accumulation into
    # sceneHdr). See vulkan/SplatPass.{hpp,cpp} and shaders/splat_common.glsl.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_project.comp"
        kSplatProjectCompSpv
        _splat_project_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_scan.comp"
        kSplatScanCompSpv
        _splat_scan_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_scan_add.comp"
        kSplatScanAddCompSpv
        _splat_scan_add_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_expand.comp"
        kSplatExpandCompSpv
        _splat_expand_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_indirect.comp"
        kSplatIndirectCompSpv
        _splat_indirect_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_radix_hist.comp"
        kSplatRadixHistCompSpv
        _splat_radix_hist_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_radix_scatter.comp"
        kSplatRadixScatterCompSpv
        _splat_radix_scatter_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_range.comp"
        kSplatRangeCompSpv
        _splat_range_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_raster.comp"
        kSplatRasterCompSpv
        _splat_raster_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_checksum.comp"
        kSplatChecksumCompSpv
        _splat_checksum_spv_h)
    # Splat VOLUME BAKE — off the frame path entirely: one scatter + one resolve
    # recorded inside SplatPass::uploadCloud's staging one-shot, voxelizing a
    # rigid cloud once into the rgba16f medium that reflection legs march.
    # Own descriptor set layout (shaders/splat_bake_common.glsl), nothing shared
    # with the raster above. See plans/splat-volume-reflections.md.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_bake_scatter.comp"
        kSplatBakeScatterCompSpv
        _splat_bake_scatter_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_bake_resolve.comp"
        kSplatBakeResolveCompSpv
        _splat_bake_resolve_spv_h)
    # Auto-exposure: log2-luma histogram over sceneHdr. See AutoExposure.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/lum_histogram.comp"
        kLumHistogramCompSpv
        _lum_histogram_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/deferred_shade.comp"
        kDeferredShadeCompSpv
        _deferred_shade_spv_h)
    # Demodulated-lighting filter + composite stages (formerly the two channels
    # of deferred_denoise.comp; shared pieces in deferred_filter_common.glsl,
    # picked up automatically by compile_vulkan_shader's *.glsl DEPENDS glob):
    #   deferred_gi_filter   — SVGF GI + soft-shadow ratio filter, recombine.
    #   deferred_refl_filter — reflection gloss reconstruction, recombine.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/deferred_gi_filter.comp"
        kDeferredGiFilterCompSpv
        _deferred_gi_filter_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/deferred_refl_filter.comp"
        kDeferredReflFilterCompSpv
        _deferred_refl_filter_spv_h)
    # Clustered light culling (deferred): per-cell light index lists consumed
    # by deferred_shade's analytic direct split. See DeferredShade::recordClusterBuild.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/cluster_build.comp"
        kClusterBuildCompSpv
        _cluster_build_spv_h)
    # Froxel volumetric lighting (deferred): inject (per-froxel in-scatter,
    # RT sun shafts + clustered lights) + integrate (front-to-back LUT).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/froxel_inject.comp"
        kFroxelInjectCompSpv
        _froxel_inject_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/froxel_integrate.comp"
        kFroxelIntegrateCompSpv
        _froxel_integrate_spv_h)
    # Particle billboard lighting (deferred): per-particle light field (sun ×
    # RT shadow, clustered lights, probe/env ambient) + camera→particle fog leg
    # for the overlay billboards. See DeferredShade::recordParticleLight.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle_light.comp"
        kParticleLightCompSpv
        _particle_light_spv_h)
    # Half-res volumetric cloud march (deferred): far-field raymarched cloud
    # deck + temporal reprojection. See DeferredShade::recordCloudMarch.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/cloud_march.comp"
        kCloudMarchCompSpv
        _cloud_march_spv_h)
    # Cloud shadow map (deferred): top-down cloud transmittance projected to the
    # ground. See DeferredShade::recordCloudShadow.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/cloud_shadow.comp"
        kCloudShadowCompSpv
        _cloud_shadow_spv_h)
    # Half-res ray-traced ambient occlusion + bent normals (deferred): short
    # cosine occlusion rays through the TLAS, temporally reprojected at half res.
    # See DeferredShade::recordRtao.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/rtao.comp"
        kRtaoCompSpv
        _rtao_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/probe_update.comp"
        kProbeUpdateCompSpv
        _probe_update_spv_h)
    # HiZ depth pyramid (occlusion-culling farthest-reduce) — see
    # HiZPyramid.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/hiz_build.comp"
        kHizBuildCompSpv
        _hiz_build_spv_h)
    # Two-phase GPU occlusion culling (phase-1 filter + AABB-vs-HiZ test) —
    # see OcclusionCull.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/occl_cull.comp"
        kOcclCullCompSpv
        _occl_cull_spv_h)
    # GPU per-instance world-matrix expansion (world = spanWorld ·
    # instanceMatrix[i]) — see InstanceExpand.{hpp,cpp} and
    # plans/gpu-driven-instances.md.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/instance_expand.comp"
        kInstanceExpandCompSpv
        _instance_expand_spv_h)
    # ParticleField density representation (plans/particle-field.md §3.3): one
    # thread per particle slot splats fixed-point sigma_t into a world-anchored
    # r32ui volume that every view's froxel pass then samples. See
    # ParticleFieldPass.{hpp,cpp} and shaders/particle_density.glsl.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle_density_scatter.comp"
        kParticleDensityScatterCompSpv
        _particle_density_scatter_spv_h)
    # r32ui -> r16f mirror: hardware trilinear for the deferred shade's
    # per-pixel dust march (view-independent, unlike the froxel LUT read).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle_density_convert.comp"
        kParticleDensityConvertCompSpv
        _particle_density_convert_spv_h)
    # ParticleField DEVICE EMITTER (Ownership::Renderer,
    # plans/particle-atmosphere.md F-C): one thread per slot evaluates a
    # stateless closed-form trajectory and writes this frame's positions AND
    # the prevPositions the motion vectors reproject through, in one dispatch.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle_emit.comp"
        kParticleEmitCompSpv
        _particle_emit_spv_h)
    # F5 surface interaction: a top-down height map of the static scene over one
    # field's footprint, one ray query per texel. The emitter solves its landing
    # against this map inside the closed form, so snow rests and rain splashes
    # with no collision pass and no per-particle state.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle_height_bake.comp"
        kParticleHeightBakeCompSpv
        _particle_height_bake_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/skinning.comp"
        kSkinningCompSpv
        _skinning_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/tet_skinning.comp"
        kTetSkinningCompSpv
        _tet_skinning_spv_h)
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
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/foam_world.comp"
        kFoamWorldCompSpv
        _foam_world_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/grass_wind.comp"
        kGrassWindCompSpv
        _grass_wind_spv_h)
    # Zero-copy vertex interop: finiteness repair over the EXPORTED position
    # buffer, the GPU stand-in for the CPU scan that guards every other BLAS
    # build (the host array is stale under interop). See VertexSanitizePipeline.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/vertex_sanitize.comp"
        kVertexSanitizeCompSpv
        _vertex_sanitize_spv_h)
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
    # ParticleField mesh representation: a SIBLING of gbuffer_indirect.vert
    # over the same render pass and the same fragment stage. It is separate
    # precisely because it breaks that file's one-instance-per-draw contract —
    # a field is one indirect draw whose instanceCount is the device-side live
    # count, so gl_InstanceIndex is the particle index there.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particlefield_gbuf.vert"
        kParticleFieldGbufVertSpv
        _particlefield_gbuf_vert_spv_h)
    # ParticleField BILLBOARD representation (plans/particle-atmosphere.md F-D):
    # a vertex-less camera-facing quad — one indirect draw of 4 vertices x the
    # device-side live count per field, corner from gl_VertexIndex and particle
    # from gl_InstanceIndex, everything else pulled by buffer_reference. Drawn
    # in the post-upscaler overlay slot, additive and unlit, so it needs neither
    # particle_light.comp nor a sort.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particlefield_billboard.vert"
        kParticleFieldBillboardVertSpv
        _particlefield_billboard_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particlefield_billboard.frag"
        kParticleFieldBillboardFragSpv
        _particlefield_billboard_frag_spv_h)
    # R8 of plans/particle-volumetric-sprites: the transmittance PREPASS. One
    # thread per slot marches the field's density volume toward the eye and
    # toward the sun and writes the pair packed into one uint, which the
    # billboard vertex stage then fetches. It used to march in that stage, four
    # times per particle (a quad has four vertices and the march origin is the
    # particle centre) — measured at +9.4 ms for 4M sprites.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particlefield_transmit.comp"
        kParticleFieldTransmitCompSpv
        _particlefield_transmit_spv_h)
    # The geometry half of T_sun: one ray query per particle toward the sun
    # against the scene TLAS, folded into the word the prepass above just
    # wrote. A separate module and not four lines in that one because a ray
    # query is a CAPABILITY — every volumetric field on every device compiles
    # the prepass, including devices with no ray query at all.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particlefield_sun_occlude.comp"
        kParticleFieldSunOccludeCompSpv
        _particlefield_sun_occlude_spv_h)
    # F4 (1): the billboard-only bloom composite. The glow-enabled fields are
    # drawn a SECOND time into a small offscreen linear-HDR target, the shared
    # bloom_down/bloom_up pyramid runs on that target alone, and this fullscreen
    # pass adds the result into the swapchain inside the same overlay
    # render-pass instance the sharp quads land in — so the billboards get a
    # bloom without the composite point moving out of the post-upscaler slot
    # that keeps them clear of TAA/DLSS/FSR entirely.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particlefield_glow.frag"
        kParticleFieldGlowFragSpv
        _particlefield_glow_frag_spv_h)
    # F4 (1), depth: the glow SOURCE target is half extent and the overlay's
    # depth is full extent, so the second draw of the glow-enabled fields has
    # nothing to depth-test against without a reduction. This fullscreen pass
    # writes that reduced buffer (compute cannot target a depth-aspect image).
    # Two variants because the overlay's depth is multisampled exactly when the
    # Canvas asked for antialiasing, and a sampler2D and a sampler2DMS are
    # different types in GLSL.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/billboard_glow_depth.frag"
        kBillboardGlowDepthFragSpv
        _billboard_glow_depth_frag_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/billboard_glow_depth.frag"
        kBillboardGlowDepthMsFragSpv
        _billboard_glow_depth_ms_frag_spv_h
        VARIANT_SUFFIX ms
        DEFINES DEPTH_MS)
    # MSAA G-buffer dominant-sample resolve (VulkanRenderer::setGbufferMsaa,
    # opt-in, default off). gbuf_resolve.comp picks the majority-covering
    # instance per pixel from the K raster samples and writes it into the
    # single-sample resolve targets every existing consumer already reads;
    # gbuf_resolve_depth.{vert,frag} is a companion fullscreen pass that
    # resolves depth (compute can't target a depth-aspect image). See
    # GbufResolve.{hpp,cpp}.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/gbuf_resolve.comp"
        kGbufResolveCompSpv
        _gbuf_resolve_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/gbuf_resolve_depth.vert"
        kGbufResolveDepthVertSpv
        _gbuf_resolve_depth_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/gbuf_resolve_depth.frag"
        kGbufResolveDepthFragSpv
        _gbuf_resolve_depth_frag_spv_h)
    # Hybrid raster TAA: post-denoise temporal resolve. Reads taaInput +
    # history + gbufMotion → writes swapchain + new history.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/taa_resolve.comp"
        kTaaResolveCompSpv
        _taa_resolve_spv_h)
    # Post-TAA RCAS sharpen (FidelityFX-style) — restores detail the temporal
    # resolve softens. See vulkan/TaaResolve.cpp (folded into that pass).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/rcas.comp"
        kRcasCompSpv
        _rcas_spv_h)
    # Post-TAA motion blur (McGuire 2012 tile-max + reconstruction), also
    # folded into vulkan/TaaResolve.cpp. Opt-in via setMotionBlur.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/motion_tilemax.comp"
        kMotionTilemaxCompSpv
        _motion_tilemax_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/motion_blur.comp"
        kMotionBlurCompSpv
        _motion_blur_spv_h)
    # FSR reactive-mask generator (G-buffer IDs flags -> R8). Shared by the
    # DLSS path as its bias-current-color mask (same deformer-flag input), so
    # compile it when either upscaler is enabled.
    if (THREEPP_WITH_FSR OR THREEPP_WITH_DLSS)
        compile_vulkan_shader(threepp
            "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/fsr_reactive.comp"
            kFsrReactiveCompSpv
            _fsr_reactive_spv_h)
    endif ()
    # Hybrid raster overlay: wireframe + Line/LineSegments draw on top of
    # the post-TAA swapchain image. Depth-tested against the G-buffer's
    # depth attachment so overlays are correctly occluded by path-traced
    # geometry. Only runs when the scene has overlay objects.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay.vert"
        kOverlayVertSpv
        _overlay_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay.frag"
        kOverlayFragSpv
        _overlay_frag_spv_h)
    # Scene-inject for the hardware-MSAA overlay pass: a fullscreen triangle
    # that seeds the multisampled overlay target with the already-composited
    # scene (copied to a 1-sample scratch) before any overlay geometry draws.
    # Only used when Canvas antialiasing > 1.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_fullscreen.vert"
        kOverlayFullscreenVertSpv
        _overlay_fullscreen_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_inject.frag"
        kOverlayInjectFragSpv
        _overlay_inject_frag_spv_h)
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
    # Splat depth stamp — the same depth buffer, written a second time from
    # the Gaussian-splat depth AOV, so the overlay is occluded by a cloud in
    # front of it (the compositor is compute and has no depth attachment).
    # Pairs with overlay_fullscreen.vert.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/splat_overlay_depth.frag"
        kSplatOverlayDepthFragSpv
        _splat_overlay_depth_frag_spv_h)
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
    # Overlay variant for Points objects — POINT_LIST topology, writes
    # gl_PointSize from the push constant, discards fragments outside
    # a unit-radius disk for round LIDAR-style dots.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_point.vert"
        kOverlayPointVertSpv
        _overlay_point_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_point.frag"
        kOverlayPointFragSpv
        _overlay_point_frag_spv_h)
    # Lit split-screen pane meshes — a secondary scissored render() draws its
    # meshes depth-tested and sun+ambient lit over a cleared background,
    # instead of the flat HUD fills (the editor's camera preview).
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_mesh_lit.vert"
        kOverlayMeshLitVertSpv
        _overlay_mesh_lit_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_mesh_lit.frag"
        kOverlayMeshLitFragSpv
        _overlay_mesh_lit_frag_spv_h)
    # Lit pane sky — fullscreen equirect background for the lit pane, sampling
    # the same envImage the deferred primary miss shows.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_pane_sky.vert"
        kOverlayPaneSkyVertSpv
        _overlay_pane_sky_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_pane_sky.frag"
        kOverlayPaneSkyFragSpv
        _overlay_pane_sky_frag_spv_h)
    # HUD sprite (TextSprite + Sprite) — alpha-blended quad with per-sprite
    # combined-image-sampler binding. Walked from the user's HUD scene during
    # the ortho overlay pass, post-PT/TAA, depth disabled.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_sprite.vert"
        kOverlaySpriteVertSpv
        _overlay_sprite_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/overlay_sprite.frag"
        kOverlaySpriteFragSpv
        _overlay_sprite_frag_spv_h)
    # World-space particle billboards (ParticleSystem). The Vulkan backend has no
    # generic ShaderMaterial path, so the particle Mesh is drawn by a dedicated
    # billboard pass in the post-TAA overlay block — depth-tested against the
    # G-buffer depth, composited onto the swapchain. Two blend variants (alpha /
    # additive) selected per material at draw time.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle.vert"
        kParticleVertSpv
        _particle_vert_spv_h)
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/particle.frag"
        kParticleFragSpv
        _particle_frag_spv_h)
    # World-space Sprite billboards (screenSpace == false) — perspective,
    # depth-tested counterpart to the ortho overlay_sprite HUD path. Shares
    # overlay_sprite.frag; only the vertex shader differs (perspective reverse-Z,
    # no GL z-remap). Drawn in the post-TAA overlay block.
    compile_vulkan_shader(threepp
        "${CMAKE_CURRENT_SOURCE_DIR}/threepp/renderers/vulkan/shaders/sprite3d.vert"
        kSprite3dVertSpv
        _sprite3d_vert_spv_h)
endif ()


# MSVC compiles one file at a time unless told otherwise. MSBuild parallelises
# across PROJECTS, and threepp is a single project of ~640 sources, so without
# /MP the whole library is built by one cl.exe serially — on the Visual Studio
# generator, which is what scikit-build-core uses for the Python wheels and what
# the IDE uses locally. Measured on 12 cores: 7.32 min -> 1.96 min (3.7x), with a
# byte-for-byte identical threepp.lib, since /MP only changes how many compiler
# processes run, not what they emit.
#
# Set as a compile OPTION, not by overriding CMAKE_CXX_FLAGS: passing
# -DCMAKE_CXX_FLAGS=/MP replaces the platform defaults and silently drops /EHsc,
# which changes the exception model and shrank the library by 11% when measured.
#
# Harmless on Ninja, which already parallelises per file — cl.exe given /MP and a
# single input just compiles it. No PCH in this build, so no interaction there.
if (MSVC)
    target_compile_options(threepp PRIVATE /MP)

    # RelWithDebInfo is MSVC's /O2 /Ob1: full optimization, but only functions
    # marked `inline` are inlined. Everything else — the non-`inline` helpers
    # that make up most of the per-frame CPU work — stays a call. Release gets
    # /Ob2 and MinSizeRel deliberately keeps /Ob1 (that is what "min size"
    # means), so only the profile config needs correcting.
    #
    # This is the config every CPU-phase A/B in this tree is measured in, so
    # leaving it at /Ob1 does not just cost time — it measures the wrong
    # inliner. The same trap cost the RLtools examples 10x until they forced
    # /Ob2 (examples/projects/RLtools/CMakeLists.txt).
    #
    # No float results move: /fp:precise on x64 has no x87 excess precision to
    # spill and the SSE2 baseline offers nothing to contract into, so an
    # inlining decision cannot change a rounded value. The cost is that
    # RelWithDebInfo stack traces get shallower.
    # Rewritten rather than appended: appending /Ob2 leaves /Ob1 on the command
    # line too, and while the last /Ob wins, cl reports D9025 for every single
    # translation unit. Editing the config flags here is directory-scoped, so it
    # reaches the library and nothing above it.
    string(REPLACE "/Ob1" "/Ob2" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    if (NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES "/Ob2")
        # A toolchain file or preset replaced the stock flags; append instead.
        string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " /Ob2")
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
