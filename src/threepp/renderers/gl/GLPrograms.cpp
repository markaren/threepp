
#include "GLPrograms.hpp"

#include "GLProgram.hpp"

#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/utils/StringUtils.hpp"

#include "threepp/renderers/shaders/ShaderLib.hpp"

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

}// namespace


GLPrograms::GLPrograms(GLBindingStates &bindingStates, GLClipping &clipping)
    : logarithmicDepthBuffer(GLCapabilities::instance().logarithmicDepthBuffer),
      floatVertexTextures(GLCapabilities::instance().floatVertexTextures),
      maxVertexUniforms(GLCapabilities::instance().maxVertexUniforms),
      vertexTextures(GLCapabilities::instance().vertexTextures),
      bindingStates(bindingStates),
      clipping(clipping) {}


int GLPrograms::getTextureEncodingFromMap(std::optional<Texture> &map) const {

    return map ? map->encoding : LinearEncoding;
}

ProgramParameters GLPrograms::getParameters(const GLRenderer &renderer, Material *material, const GLLights::LightState &lights, int numShadows, Scene *scene, Object3D *object) {

    auto mapMaterial = dynamic_cast<MaterialWithMap*>(material);
    auto alphaMaterial = dynamic_cast<MaterialWithAlphaMap*>(material);
    auto aomapMaterial = dynamic_cast<MaterialWithAoMap*>(material);
    auto bumpmapMaterial = dynamic_cast<MaterialWithBumpMap*>(material);
    auto matcapMaterial = dynamic_cast<MaterialWithMatCap*>(material);
    auto envmapMaterial = dynamic_cast<MaterialWithEnvMap*>(material);
    auto lightmapMaterial = dynamic_cast<MaterialWithLightMap*>(material);
    auto emissiveMaterial = dynamic_cast<MaterialWithEmissive*>(material);
    auto normalMaterial = dynamic_cast<MaterialWithNormalMap*>(material);
    auto specularMapMaterial = dynamic_cast<MaterialWithSpecularMap*>(material);

    std::optional<Texture> emptyMap;

    ProgramParameters p;

    p.shaderID = shaderIDs[material->type()];
    p.shaderName = material->type();

    p.isRawShaderMaterial = instanceof <RawShaderMaterial>(material);

    auto instancedMesh = dynamic_cast<InstancedMesh *>(object);
    p.instancing = instancedMesh != nullptr;
    p.instancingColor = instancedMesh != nullptr && instancedMesh->instanceColor != nullptr;

    p.supportsVertexTextures = vertexTextures;

    p.map = mapMaterial == nullptr ? false : mapMaterial->map.has_value();
    p.mapEncoding = getTextureEncodingFromMap(mapMaterial == nullptr ? emptyMap : mapMaterial->map);
    p.matcap = matcapMaterial == nullptr ? false : matcapMaterial->matcap.has_value();
    p.matcapEncoding = getTextureEncodingFromMap(matcapMaterial == nullptr ? emptyMap : matcapMaterial->matcap);
    p.envMap = envmapMaterial == nullptr ? false : envmapMaterial->envMap.has_value();
    p.envMapMode = p.envMap && envmapMaterial->envMap->mapping.has_value();
    p.envMapEncoding = getTextureEncodingFromMap(envmapMaterial == nullptr ? emptyMap : envmapMaterial->envMap);
    p.envMapCubeUV = p.envMapMode ? (envmapMaterial->envMap->mapping.value_or(-1) == CubeReflectionMapping || envmapMaterial->envMap->mapping.value_or(-1) == CubeRefractionMapping) : false;
    p.lightMap = lightmapMaterial == nullptr ? false : lightmapMaterial->lightMap.has_value();
    p.lightMapEncoding = getTextureEncodingFromMap(lightmapMaterial == nullptr ? emptyMap : envmapMaterial->envMap);
    p.aoMap = aomapMaterial == nullptr ? false : aomapMaterial->aoMap.has_value();
    p.emissiveMap = emissiveMaterial == nullptr ? false : emissiveMaterial->emissiveMap.has_value();
    p.emissiveMapEncoding = getTextureEncodingFromMap(emissiveMaterial == nullptr ? emptyMap : emissiveMaterial->emissiveMap);
    p.bumpMap = bumpmapMaterial == nullptr ? false : bumpmapMaterial->bumpMap.has_value();
    p.normalMap = normalMaterial == nullptr ? false : normalMaterial->normalMap.has_value();
    p.objectSpaceNormalMap = normalMaterial != nullptr && normalMaterial->normalMapType == ObjectSpaceNormalMap;
    p.tangentSpaceNormalMap = normalMaterial != nullptr && normalMaterial->normalMapType == TangentSpaceNormalMap;
//    clearcoatMap: !! material.clearcoatMap
//    clearcoatRoughnessMap: !! material.clearcoatRoughnessMap
//    clearcoatNormalMap: !! material.clearcoatNormalMap
//    displacementMap: !! material.displacementMap
//    roughnessMap: !! material.roughnessMap
//    metalnessMap: !! material.metalnessMap
    p.specularMap = specularMapMaterial == nullptr ? false : specularMapMaterial->specularMap.has_value();
    p.alphaMap = alphaMaterial == nullptr ? false : alphaMaterial->alphaMap.has_value();

//    gradientMap: !! material.gradientMap

//    sheen: !! material.sheen
//
//    transmission: !! material.transmission
//    transmissionMap: !! material.transmissionMap
//    thicknessMap: !! material.thicknessMap
//
//    combine: material.combine

    p.numDirLights = (int) lights.directional.size();
    p.numPointLights = (int) lights.point.size();
    p.numSpotLights = (int) lights.spot.size();
    p.numRectAreaLights = 0;
    p.numHemiLights = 0;

    p.numDirLightShadows = (int) lights.directionalShadowMap.size();
    p.numPointLightShadows = (int) lights.pointShadowMap.size();
    p.numSpotLightShadows = (int) lights.spotShadowMap.size();

    p.numClippingPlanes = clipping.numPlanes;
    p.numClipIntersection = clipping.numIntersection;

    p.dithering = material->dithering;

    p.shadowMapEnabled = renderer.shadowMap.enabled && numShadows > 0;
    p.shadowMapType = renderer.shadowMap.type;

    p.toneMapping = material->toneMapped ? renderer.toneMapping : NoToneMapping;
    p.physicallyCorrectLights = renderer.physicallyCorrectLights;

    p.premultipliedAlpha = material->premultipliedAlpha;

    p.alphaTest = material->alphaTest;
    p.doubleSided = material->side == DoubleSide;
    p.flipSided = material->side == BackSide;

    p.index0AttributeName = std::nullopt;

    return p;
}

std::string GLPrograms::getProgramCacheKey(const GLRenderer &renderer, const ProgramParameters &parameters) {

    std::vector<std::string> array;

    if (parameters.shaderID) {

        array.emplace_back(*parameters.shaderID);

    } else {

        array.emplace_back(parameters.fragmentShader);
        array.emplace_back(parameters.vertexShader);
    }

    if (!parameters.defines.empty()) {

        for (const auto &[name, value] : parameters.defines) {

            array.emplace_back(name);
            array.emplace_back(value);
        }
    }

    if (!parameters.isRawShaderMaterial) {

        auto hash = utils::split(parameters.hash(), '\n');
        for (const auto &value : hash) {

            array.emplace_back(value);
        }

        array.emplace_back(std::to_string(renderer.outputEncoding));
        array.emplace_back(std::to_string(renderer.gammaFactor));
    }

    //    array.emplace_back(parameters.customProgramCacheKey);

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

std::shared_ptr<GLProgram> GLPrograms::acquireProgram(const GLRenderer &renderer, const ProgramParameters &parameters, const std::string &cacheKey) {

    std::shared_ptr<GLProgram> program = nullptr;

    // Check if code has been already compiled
    for (auto &preexistingProgram : programs) {

        if (preexistingProgram->cacheKey == cacheKey) {

            program = preexistingProgram;
            ++program->usedTimes;

            break;
        }
    }

    if (!program) {

        program = GLProgram::create(renderer, cacheKey, parameters, bindingStates );
        programs.emplace_back(program);
    }

    return program;
}

void GLPrograms::releaseProgram(std::shared_ptr<GLProgram> &program) {

    if (--program->usedTimes == 0) {

        auto it = find(programs.begin(), programs.end(), program);
        auto i = it - programs.begin();

        // Remove from unordered set
        programs[i] = programs[programs.size() - 1];
        programs.pop_back();

        // Free WebGL resources
        program->destroy();
    }
}
