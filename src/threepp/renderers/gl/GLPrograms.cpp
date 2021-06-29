
#include "GLPrograms.hpp"

#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/utils/InstanceOf.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "threepp/renderers/shaders/ShaderLib.hpp"

#include <glad/glad.h>


using namespace threepp;
using namespace threepp::gl;

namespace {

    std::unordered_map<std::string, std::string> shaderIDs{
            {"MeshDepthMaterial", "depth"},
            {"MeshDistanceMaterial", "distanceRGBA"},
            {"MeshNormalMaterial", "normal"},
            {"MeshBasicMaterial", "basic"},
            {"MeshLambertMaterial", "lambert"},
            {"MeshPhongMaterial", "phong"},
            {"MeshToonMaterial", "toon"},
            {"MeshStandardMaterial", "physical"},
            {"MeshPhysicalMaterial", "physical"},
            {"MeshMatcapMaterial", "matcap"},
            {"LineBasicMaterial", "basic"},
            {"LineDashedMaterial", "dashed"},
            {"PointsMaterial", "points"},
            {"ShadowMaterial", "shadow"},
            {"SpriteMaterial", "sprite"}};

    //    // clang-format off
    //    const std::vector<std::string> parameterNames {
    //            "supportsVertexTextures", "outputEncoding", "instancing", "instancingColor",
    //            "map", "mapEncoding", "matcap", "matcapEncoding", "envMap", "envMapMode", "envMapEncoding", "envMapCubeUV",
    //            "lightMap", "lightMapEncoding", "aoMap", "emissiveMap", "emissiveMapEncoding", "bumpMap", "normalMap", "objectSpaceNormalMap", "tangentSpaceNormalMap", "clearcoatMap", "clearcoatRoughnessMap", "clearcoatNormalMap", "displacementMap", "specularMap",
    //            "roughnessMap", "metalnessMap", "gradientMap",
    //            "alphaMap", "combine", "vertexColors", "vertexAlphas", "vertexTangents", "vertexUvs", "uvsVertexOnly", "fog", "useFog", "fogExp2",
    //            "flatShading", "sizeAttenuation", "logarithmicDepthBuffer",
    //            "maxBones", "useVertexTexture", "morphTargets", "morphNormals", "premultipliedAlpha",
    //            "numDirLights", "numPointLights", "numSpotLights", "numHemiLights", "numRectAreaLights",
    //            "numDirLightShadows", "numPointLightShadows", "numSpotLightShadows",
    //            "shadowMapEnabled", "shadowMapType", "toneMapping", "physicallyCorrectLights",
    //            "alphaTest", "doubleSided", "flipSided", "numClippingPlanes", "numClipIntersection", "depthPacking", "dithering",
    //    };
    //    // clang-format on
}// namespace


GLPrograms::Parameters::Parameters(
        const GLPrograms &scope,
        Material *material,
        const GLLights::LightState &lights,
        int numShadows,
        Scene *scene,
        Object3D *object) {

    shaderID = shaderIDs[material->type()];
    shaderName = material->type();

    isRawShaderMaterial = instanceof <RawShaderMaterial>(material);

    supportsVertexTextures = scope.vertexTextures;

    numDirLights = (int) lights.directional.size();
    numPointLights = (int) lights.point.size();
    numSpotLights = (int) lights.spot.size();
    numRectAreaLights = 0;
    numHemiLights = 0;
}

std::string GLPrograms::Parameters::hash() const {

    std::string s;

    s += std::to_string(instancing) + '\n';
    s += std::to_string(instancingColor) + '\n';

    s += std::to_string(supportsVertexTextures) + '\n';
    s += std::to_string(outputEncoding) + '\n';
    s += std::to_string(map) + '\n';
    s += std::to_string(mapEncoding) + '\n';
    s += std::to_string(matcap) + '\n';
    s += std::to_string(matcapEncoding) + '\n';
    s += std::to_string(envMap) + '\n';
    s += std::to_string(envMapEncoding) + '\n';
    s += std::to_string(envMapMode) + '\n';
    s += std::to_string(envMapEncoding) + '\n';
    s += std::to_string(envMapCubeUV) + '\n';
    s += std::to_string(lightMap) + '\n';
    s += std::to_string(lightMapEncoding) + '\n';
    s += std::to_string(aoMap) + '\n';
    s += std::to_string(emissiveMap) + '\n';
    s += std::to_string(emissiveMapEncoding) + '\n';
    s += std::to_string(bumpMap) + '\n';
    s += std::to_string(normalMap) + '\n';
    s += std::to_string(objectSpaceNormalMap) + '\n';
    s += std::to_string(tangentSpaceNormalMap) + '\n';
    s += std::to_string(clearcoatMap) + '\n';
    s += std::to_string(clearcoatRoughnessMap) + '\n';
    s += std::to_string(clearcoatNormalMap) + '\n';
    s += std::to_string(displacementMap) + '\n';
    s += std::to_string(roughnessMap) + '\n';
    s += std::to_string(metalnessMap) + '\n';
    s += std::to_string(specularMap) + '\n';
    s += std::to_string(alphaMap) + '\n';

    s += std::to_string(gradientMap) + '\n';

    s += std::to_string(sheen) + '\n';

    s += std::to_string(transmission) + '\n';
    s += std::to_string(transmissionMap) + '\n';
    s += std::to_string(thicknessMap) + '\n';

    s += std::to_string(combine) + '\n';

    s += std::to_string(vertexTangents) + '\n';
    s += std::to_string(vertexColors) + '\n';
    s += std::to_string(vertexAlphas) + '\n';
    s += std::to_string(vertexUvs) + '\n';
    s += std::to_string(uvsVertexOnly) + '\n';

    s += std::to_string(fog) + '\n';
    s += std::to_string(useFog) + '\n';
    s += std::to_string(fogExp2) + '\n';

    s += std::to_string(flatShading) + '\n';

    s += std::to_string(sizeAttenuation) + '\n';
    s += std::to_string(logarithmicDepthBuffer) + '\n';

    s += std::to_string(numDirLights) + '\n';
    s += std::to_string(numPointLights) + '\n';
    s += std::to_string(numSpotLights) + '\n';
    s += std::to_string(numRectAreaLights) + '\n';
    s += std::to_string(numHemiLights) + '\n';

    s += std::to_string(numDirLightShadows) + '\n';
    s += std::to_string(numPointLightShadows) + '\n';
    s += std::to_string(numSpotLightShadows) + '\n';

    s += std::to_string(numClippingPlanes) + '\n';
    s += std::to_string(numClipIntersection) + '\n';

    s += std::to_string(dithering) + '\n';

    s += std::to_string(shadowMapEnabled) + '\n';
    s += std::to_string(shadowMapType) + '\n';

    s += std::to_string(toneMapping) + '\n';
    s += std::to_string(physicallyCorrectLights) + '\n';

    s += std::to_string(premultipliedAlpha) + '\n';

    s += std::to_string(alphaTest) + '\n';
    s += std::to_string(doubleSided) + '\n';
    s += std::to_string(flipSided) + '\n';

    s += std::to_string(depthPacking) + '\n';

    return s;
}

GLPrograms::GLPrograms()
    : logarithmicDepthBuffer(GLCapabilities::instance().logarithmicDepthBuffer),
      floatVertexTextures(GLCapabilities::instance().floatVertexTextures),
      maxVertexUniforms(GLCapabilities::instance().maxVertexUniforms),
      vertexTextures(GLCapabilities::instance().vertexTextures) {}


int GLPrograms::getTextureEncodingFromMap(std::optional<Texture> &map) const {

    int encoding;

    if (map) {

        encoding = map->encoding;

    } else {

        encoding = LinearEncoding;
    }

    return encoding;
}

GLPrograms::Parameters GLPrograms::getParameters(Material *material, const GLLights::LightState &lights, int numShadows, Scene *scene, Object3D *object) {

    return GLPrograms::Parameters(*this, material, lights, numShadows, scene, object);
}

std::string GLPrograms::getProgramCacheKey(const GLRenderer &renderer, const GLPrograms::Parameters &parameters) {

    std::vector<std::string> array;

    if (parameters.shaderID) {

        array.emplace_back(*parameters.shaderID);

    } else {

        array.emplace_back(parameters.fragmentShader);
        array.emplace_back(parameters.vertexShader);
    }

    if (parameters.defines) {

        for (const auto &[name, value] : *parameters.defines) {

            array.emplace_back(name);
            array.emplace_back(value);
        }
    }

    if (!parameters.isRawShaderMaterial) {

        auto hash = utils::split(parameters.hash(), '\n');
        for (const auto& value : hash) {

            array.emplace_back(value);
        }

        array.emplace_back(std::to_string(renderer.outputEncoding));
        array.emplace_back(std::to_string(renderer.gammaFactor));
    }

    array.emplace_back(parameters.customProgramCacheKey);

    return utils::join(array, '\n');
}

std::unordered_map<std::string, Uniform> GLPrograms::getUniforms(Material *material) {

    std::unordered_map<std::string, Uniform> uniforms;

    if (shaderIDs.count(material->type())) {

        auto shaderID = shaderIDs.at(material->type());

        auto shader = shaders::ShaderLib::instance().get(shaderID);
        uniforms = shader.uniforms;

    } else {

        uniforms = material->uniforms;
    }

    return uniforms;
}
