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
        "threepp/core/Shader.hpp"
        "threepp/core/Uniform.hpp"

        "threepp/cameras/Camera.hpp"
        "threepp/cameras/PerspectiveCamera.hpp"
        "threepp/cameras/OrthographicCamera.hpp"
        "threepp/cameras/CameraView.hpp"

        "threepp/extras/core/Curve.hpp"
        "threepp/extras/curve/CatmullRomCurve3.hpp"

        "threepp/input/KeyListener.hpp"
        "threepp/input/MouseListener.hpp"

        "threepp/geometries/geometries.hpp"
        "threepp/geometries/BoxGeometry.hpp"
        "threepp/geometries/CircleGeometry.hpp"
        "threepp/geometries/ConeGeometry.hpp"
        "threepp/geometries/CylinderGeometry.hpp"
        "threepp/geometries/PlaneGeometry.hpp"
        "threepp/geometries/SphereGeometry.hpp"
        "threepp/geometries/TubeGeometry.hpp"

        "threepp/helpers/helpers.hpp"
        "threepp/helpers/AxesHelper.hpp"
        "threepp/helpers/ArrowHelper.hpp"
        "threepp/helpers/Box3Helper.hpp"
        "threepp/helpers/CameraHelper.hpp"
        "threepp/helpers/DirectionalLightHelper.hpp"
        "threepp/helpers/GridHelper.hpp"
        "threepp/helpers/PlaneHelper.hpp"
        "threepp/helpers/PointLightHelper.hpp"
        "threepp/helpers/SpotLightHelper.hpp"

        "threepp/loaders/loaders.hpp"
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
        "threepp/objects/Reflector.hpp"
        "threepp/objects/Water.hpp"

        "threepp/textures/DepthTexture.hpp"
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
        "threepp/lights/SpotLightShadow.hpp"

        "threepp/renderers/GLRenderer.hpp"
        "threepp/renderers/GLRenderTarget.hpp"

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
        "threepp/renderers/gl/GLProperties.hpp"
        "threepp/renderers/gl/GLProgram.hpp"
        "threepp/renderers/gl/GLPrograms.hpp"
        "threepp/renderers/gl/GLRenderLists.hpp"
        "threepp/renderers/gl/GLRenderStates.hpp"
        "threepp/renderers/gl/GLShadowMap.hpp"
        "threepp/renderers/gl/GLState.hpp"
        "threepp/renderers/gl/GLTextures.hpp"
        "threepp/renderers/gl/GLUniforms.hpp"

        "threepp/renderers/shaders/ShaderChunk.hpp"
        "threepp/renderers/shaders/ShaderLib.hpp"
        "threepp/renderers/shaders/UniformsLib.hpp"

        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
    list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

set(privateHeaders

        "threepp/renderers/gl/glHelper.hpp"
        "threepp/renderers/gl/GLUtils.hpp"
        "threepp/renderers/gl/UniformUtils.hpp"

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
        "threepp/loaders/MTLLoader.cpp"
        "threepp/loaders/OBJLoader.cpp"
        "threepp/loaders/STLLoader.cpp"
        "threepp/loaders/TextureLoader.cpp"

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
        "threepp/lights/SpotLight.cpp"
        "threepp/lights/SpotLightShadow.cpp"

        "threepp/objects/Mesh.cpp"
        "threepp/objects/Reflector.cpp"
        "threepp/objects/Water.cpp"

        "threepp/textures/Texture.cpp"

        "threepp/utils/uuid.cpp"

        "threepp/renderers/GLRenderer.cpp"

        "threepp/renderers/gl/GLAttributes.cpp"
        "threepp/renderers/gl/GLBackground.cpp"
        "threepp/renderers/gl/GLBindingStates.cpp"
        "threepp/renderers/gl/GLBufferRenderer.cpp"
        "threepp/renderers/gl/GLCapabilities.cpp"
        "threepp/renderers/gl/GLGeometries.cpp"
        "threepp/renderers/gl/GLInfo.cpp"
        "threepp/renderers/gl/GLLights.cpp"
        "threepp/renderers/gl/GLObjects.cpp"
        "threepp/renderers/gl/GLProgram.cpp"
        "threepp/renderers/gl/GLPrograms.cpp"
        "threepp/renderers/gl/GLMaterials.cpp"
        "threepp/renderers/gl/GLShadowMap.cpp"
        "threepp/renderers/gl/GLState.cpp"
        "threepp/renderers/gl/GLTextures.cpp"
        "threepp/renderers/gl/GLUniforms.cpp"
        "threepp/renderers/gl/GLRenderLists.cpp"
        "threepp/renderers/gl/ProgramParameters.cpp"

        "threepp/renderers/shaders/ShaderLib.cpp"

        )

if (assimp_FOUND)
  list(APPEND sources "threepp/loaders/AssimpLoader.cpp")
endif()


set(generatedSourcesDir "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generatedSourcesDir}")

set(THREEPP_SHADER_INCLUDES)
set(THREEPP_SHADERCHUNK_CODE)
set(THREEPP_SHADERLIB_CODE)

file(GLOB files "${CMAKE_SOURCE_DIR}/data/shaders/ShaderChunk/*.glsl")
foreach(shaderFile ${files})
    get_filename_component(fileName ${shaderFile} NAME_WLE)
    set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk/${fileName}.hpp")
    file(READ ${shaderFile} text)
    set(text "\
namespace threepp::shaders::shaderchunk {\n\n\
const char* ${fileName}=R\"(${text})\";\n\n\
}\n"
        )
    file(WRITE "${header_file}"
            "#ifndef THREEPP_${fileName}_HPP\n"
            "#define THREEPP_${fileName}_HPP\n\n"
            )
    file(APPEND "${header_file}" "${text}")
    file(APPEND "${header_file}" "\n\n#endif")

    set(THREEPP_SHADER_INCLUDES "${THREEPP_SHADER_INCLUDES}\n#include \"ShaderChunk/${fileName}.hpp\"")

    set(THREEPP_SHADERCHUNK_CODE
            "${THREEPP_SHADERCHUNK_CODE}\tdata_[\"${fileName}\"] = shaderchunk::${fileName};\n"
    )

endforeach()

file(GLOB files "${CMAKE_SOURCE_DIR}/data/shaders/ShaderLib/*.glsl")
foreach(shaderFile ${files})
    get_filename_component(fileName ${shaderFile} NAME_WLE)
    set(header_file "${generatedSourcesDir}/threepp/renderers/shaders/ShaderLib/${fileName}.hpp")
    file(READ ${shaderFile} text)
    set(text "\
namespace threepp::shaders::shaderlib {\n\n\
const char* ${fileName}=R\"(${text})\";\n\n\
}\n"
            )
    file(WRITE "${header_file}"
            "#ifndef THREEPP_${fileName}_HPP\n"
            "#define THREEPP_${fileName}_HPP\n\n"
            )
    file(APPEND "${header_file}" "${text}")
    file(APPEND "${header_file}" "\n\n#endif")

    set(THREEPP_SHADER_INCLUDES "${THREEPP_SHADER_INCLUDES}\n#include \"ShaderLib/${fileName}.hpp\"")

    set(THREEPP_SHADERLIB_CODE
            "${THREEPP_SHADERLIB_CODE}\tdata_[\"${fileName}\"] = shaderlib::${fileName};\n"
            )

endforeach()

configure_file(
        "threepp/renderers/shaders/ShaderChunk.cpp.in"
        "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp"
        @ONLY
)


add_library(threepp ${sources} ${privateHeaders} ${publicHeadersFull} "${generatedSourcesDir}/threepp/renderers/shaders/ShaderChunk.cpp")
add_library(threepp::threepp ALIAS threepp)
target_compile_features(threepp PUBLIC cxx_std_17)
target_link_libraries(threepp
        PRIVATE
        glad::glad
        glfw::glfw
        stb::stb)
target_include_directories(threepp
        PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        PRIVATE
        "${generatedSourcesDir}"
        "${CMAKE_CURRENT_SOURCE_DIR}")

if (assimp_FOUND)
  target_link_libraries(threepp PRIVATE assimp::assimp)
endif()
