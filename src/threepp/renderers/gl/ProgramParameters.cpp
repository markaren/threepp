
#include "threepp/renderers/gl/ProgramParameters.hpp"

#include "threepp/renderers/Renderer.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <sstream>

using namespace threepp;
using namespace threepp::gl;

namespace {

    ColorSpace getTextureEncodingFromMap(const std::shared_ptr<Texture>& map) {

        return map ? map->colorSpace : ColorSpace::Linear;
    }

}// namespace

ProgramParameters::ProgramParameters(
        const Renderer& renderer,
        const ShadowConfig& shadowConfig,
        const RendererCapabilities& capabilities,
        const GLClipping& clipping,
        const Lights::LightState& lights,
        size_t numShadows,
        Object3D* object,
        Scene* scene,
        Material* material,
        Texture* resolvedEnvMap,
        const std::unordered_map<std::string, std::string>& shaderIDs) {

    auto mapMaterial = dynamic_cast<MaterialWithMap*>(material);
    auto alphaMaterial = dynamic_cast<MaterialWithAlphaMap*>(material);
    auto aomapMaterial = dynamic_cast<MaterialWithAoMap*>(material);
    auto bumpmapMaterial = dynamic_cast<MaterialWithBumpMap*>(material);
    auto matcapMaterial = dynamic_cast<MaterialWithMatCap*>(material);
    auto gradientMaterial = dynamic_cast<MaterialWithGradientMap*>(material);
    auto envmapMaterial = dynamic_cast<MaterialWithEnvMap*>(material);
    auto lightmapMaterial = dynamic_cast<MaterialWithLightMap*>(material);
    auto emissiveMaterial = dynamic_cast<MaterialWithEmissive*>(material);
    auto normalMaterial = dynamic_cast<MaterialWithNormalMap*>(material);
    auto specularMapMaterial = dynamic_cast<MaterialWithSpecularMap*>(material);
    auto displacementMapMaterial = dynamic_cast<MaterialWithDisplacementMap*>(material);
    auto combineMaterial = dynamic_cast<MaterialWithCombine*>(material);
    auto flatshadeMaterial = dynamic_cast<MaterialWithFlatShading*>(material);
    auto vertextangentsMaterial = dynamic_cast<MaterialWithVertexTangents*>(material);
    auto depthpackMaterial = dynamic_cast<MaterialWithDepthPacking*>(material);
    auto sheenMaterial = dynamic_cast<MaterialWithSheen*>(material);
    auto shaderMaterial = dynamic_cast<ShaderMaterial*>(material);
    auto definesMaterial = dynamic_cast<MaterialWithDefines*>(material);
    auto thicknessMaterial = dynamic_cast<MaterialWithThickness*>(material);
    auto clearcoatMaterial = dynamic_cast<MaterialWithClearcoat*>(material);
    auto transmissionMaterial = dynamic_cast<MaterialWithTransmission*>(material);
    auto roughnessMaterial = dynamic_cast<MaterialWithRoughness*>(material);
    auto metallnessMaterial = dynamic_cast<MaterialWithMetalness*>(material);

    std::string vShader, fShader;
    if (shaderIDs.contains(material->type())) {

        shaderID = shaderIDs.at(material->type());
        const auto shader = shaders::ShaderLib::instance().get(*shaderID);
        vShader = shader.vertexShader;
        fShader = shader.fragmentShader;

    } else {

        vShader = shaderMaterial->vertexShader;
        fShader = shaderMaterial->fragmentShader;
    }

    shaderName = material->type();

    vertexShader = vShader;
    fragmentShader = fShader;

    if (definesMaterial) {
        defines = definesMaterial->defines;
    }

    isRawShaderMaterial = material->is<RawShaderMaterial>();

    precision = "highp";

    auto instancedMesh = dynamic_cast<InstancedMesh*>(object);
    instancing = instancedMesh != nullptr;
    instancingColor = instancedMesh != nullptr && instancedMesh->instanceColor() != nullptr;

    supportsVertexTextures = capabilities.vertexTextures;
    outputEncoding = renderer.outputColorSpace;

    map = mapMaterial && mapMaterial->map;
    mapEncoding = getTextureEncodingFromMap(map ? mapMaterial->map : nullptr);
    matcap = matcapMaterial && matcapMaterial->matcap;
    matcapEncoding = getTextureEncodingFromMap(matcap ? matcapMaterial->matcap : nullptr);
    // Pure three.js port: WebGLPrograms calls `cubeuvmaps.get( material.envMap || environment )`
    // and reads `.mapping` from the *resolved* texture (PMREM atlas for equirect sources).
    // The caller threads the resolved envMap in via `resolvedEnvMap`.
    Texture* effectiveEnvMap = resolvedEnvMap;

    envMap = effectiveEnvMap != nullptr;
    if (envMap) {
        envMapMode = as_integer(effectiveEnvMap->mapping);
    }
    envMapEncoding = effectiveEnvMap ? effectiveEnvMap->colorSpace : ColorSpace::Linear;
    envMapCubeUV = envMapMode != 0 &&
                   (static_cast<Mapping>(envMapMode) == Mapping::CubeUVReflection ||
                    static_cast<Mapping>(envMapMode) == Mapping::CubeUVRefraction);
    lightMap = lightmapMaterial && lightmapMaterial->lightMap;
    lightMapEncoding = getTextureEncodingFromMap(lightMap ? lightmapMaterial->lightMap : nullptr);
    aoMap = aomapMaterial && aomapMaterial->aoMap;
    emissiveMap = emissiveMaterial && emissiveMaterial->emissiveMap;
    emissiveMapEncoding = getTextureEncodingFromMap(emissiveMap ? emissiveMaterial->emissiveMap : nullptr);
    bumpMap = bumpmapMaterial && bumpmapMaterial->bumpMap;
    normalMap = normalMaterial && normalMaterial->normalMap;
    objectSpaceNormalMap = normalMaterial && normalMaterial->normalMapType == NormalMapType::ObjectSpace;
    tangentSpaceNormalMap = normalMaterial && normalMaterial->normalMapType == NormalMapType::TangentSpace;
    clearcoatMap = clearcoatMaterial && clearcoatMaterial->clearcoatMap;
    clearcoatRoughnessMap = clearcoatMaterial && clearcoatMaterial->clearcoatRoughnessMap;
    clearcoatNormalMap = clearcoatMaterial && clearcoatMaterial->clearcoatNormalMap;
    displacementMap = displacementMapMaterial && displacementMapMaterial->displacementMap;
    roughnessMap = roughnessMaterial && roughnessMaterial->roughnessMap;
    metalnessMap = metallnessMaterial && metallnessMaterial->metalnessMap;
    specularMap = specularMapMaterial && specularMapMaterial->specularMap;
    alphaMap = alphaMaterial && alphaMaterial->alphaMap;

    gradientMap = gradientMaterial && gradientMaterial->gradientMap;

    if (sheenMaterial) {
        sheen = sheenMaterial->sheen;
    }

    transmission = transmissionMaterial && transmissionMaterial->transmission > 0;
    transmissionMap = transmissionMaterial && transmissionMaterial->transmissionMap;
    thicknessMap = thicknessMaterial && thicknessMaterial->thicknessMap;

    if (combineMaterial) {
        combine = combineMaterial->combine;
    }

    vertexTangents = normalMaterial && vertextangentsMaterial && vertextangentsMaterial->vertexTangents;
    vertexColors = material->vertexColors;
    vertexAlphas = material->vertexColors &&
                   object->geometry() &&
                   object->geometry()->hasAttribute("color") &&
                   object->geometry()->getAttribute<float>("color")->itemSize() == 4;
    vertexUvs = true;     // TODO
    uvsVertexOnly = false;// TODO;

    fog = scene->fog.has_value();
    useFog = material->fog;
    fogExp2 = scene->fog.has_value() && std::holds_alternative<FogExp2>(*scene->fog);

    if (flatshadeMaterial) {
        flatShading = flatshadeMaterial->flatShading;
    }

    auto sizeMaterial = material->as<MaterialWithSize>();
    sizeAttenuation = sizeMaterial ? sizeMaterial->sizeAttenuation : false;

    skinning = object->is<SkinnedMesh>();
    maxBones = 64;// TODO
    useVertexTexture = capabilities.floatVertexTextures;

    if (auto m = material->as<MaterialWithMorphTargets>()) {
        morphTargets = m->morphTargets;
        morphNormals = m->morphNormals;
    }

    numDirLights = lights.directional.size();
    numPointLights = lights.point.size();
    numSpotLights = lights.spot.size();
    numRectAreaLights = lights.rectArea.size();
    numHemiLights = lights.hemi.size();

    numDirLightShadows = lights.directionalShadowMap.size();
    numPointLightShadows = lights.pointShadowMap.size();
    numSpotLightShadows = lights.spotShadowMap.size();

    numClippingPlanes = clipping.numPlanes;
    numClipIntersection = clipping.numIntersection;

    dithering = material->dithering;

    shadowMapEnabled = shadowConfig.enabled && numShadows > 0;
    shadowMapType = shadowConfig.type;

    toneMapping = material->toneMapped ? renderer.toneMapping : ToneMapping::None;
    useLegacyLights = renderer.useLegacyLights;

    premultipliedAlpha = material->premultipliedAlpha;

    alphaTest = material->alphaTest;
    doubleSided = material->side == Side::Double;
    flipSided = material->side == Side::Back;

    depthPacking = depthpackMaterial ? as_integer(depthpackMaterial->depthPacking) : 0;

    if (shaderMaterial) {
        index0AttributeName = shaderMaterial->index0AttributeName;
    }
}

std::string ProgramParameters::hash() const {

    std::stringstream s;

    s << std::to_string(instancing) << '\n';
    s << std::to_string(instancingColor) << '\n';

    s << std::to_string(supportsVertexTextures) << '\n';
    s << std::to_string(as_integer(outputEncoding)) << '\n';
    s << std::to_string(map) << '\n';
    s << std::to_string(as_integer(mapEncoding)) << '\n';
    s << std::to_string(matcap) << '\n';
    s << std::to_string(as_integer(matcapEncoding)) << '\n';
    s << std::to_string(envMap) << '\n';
    s << std::to_string(as_integer(envMapEncoding)) << '\n';
    s << std::to_string(envMapMode) << '\n';
    s << std::to_string(as_integer(envMapEncoding)) << '\n';
    s << std::to_string(envMapCubeUV) << '\n';
    s << std::to_string(lightMap) << '\n';
    s << std::to_string(as_integer(lightMapEncoding)) << '\n';
    s << std::to_string(aoMap) << '\n';
    s << std::to_string(emissiveMap) << '\n';
    s << std::to_string(as_integer(emissiveMapEncoding)) << '\n';
    s << std::to_string(bumpMap) << '\n';
    s << std::to_string(normalMap) << '\n';
    s << std::to_string(objectSpaceNormalMap) << '\n';
    s << std::to_string(tangentSpaceNormalMap) << '\n';
    s << std::to_string(clearcoatMap) << '\n';
    s << std::to_string(clearcoatRoughnessMap) << '\n';
    s << std::to_string(clearcoatNormalMap) << '\n';
    s << std::to_string(displacementMap) << '\n';
    s << std::to_string(roughnessMap) << '\n';
    s << std::to_string(metalnessMap) << '\n';
    s << std::to_string(specularMap) << '\n';
    s << std::to_string(alphaMap) << '\n';

    s << std::to_string(gradientMap) << '\n';

    if (sheen.has_value()) {
        s << *sheen << '\n';
    } else {
        s << "undefined \n";
    }

    s << std::to_string(transmission) << '\n';
    s << std::to_string(transmissionMap) << '\n';
    s << std::to_string(thicknessMap) << '\n';

    s << (combine.has_value() ? std::to_string(as_integer(*combine)) : std::string("undefined")) << '\n';

    s << std::to_string(vertexTangents) << '\n';
    s << std::to_string(vertexColors) << '\n';
    s << std::to_string(vertexAlphas) << '\n';
    s << std::to_string(vertexUvs) << '\n';
    s << std::to_string(uvsVertexOnly) << '\n';

    s << std::to_string(fog) << '\n';
    s << std::to_string(useFog) << '\n';
    s << std::to_string(fogExp2) << '\n';

    s << std::to_string(flatShading) << '\n';

    s << std::to_string(sizeAttenuation) << '\n';
    s << std::to_string(logarithmicDepthBuffer) << '\n';

    s << std::to_string(morphTargets) << '\n';
    s << std::to_string(morphNormals) << '\n';

    s << std::to_string(skinning) << '\n';
    s << std::to_string(useVertexTexture) << '\n';

    s << std::to_string(numDirLights) << '\n';
    s << std::to_string(numPointLights) << '\n';
    s << std::to_string(numSpotLights) << '\n';
    s << std::to_string(numRectAreaLights) << '\n';
    s << std::to_string(numHemiLights) << '\n';

    s << std::to_string(numDirLightShadows) << '\n';
    s << std::to_string(numPointLightShadows) << '\n';
    s << std::to_string(numSpotLightShadows) << '\n';

    s << std::to_string(numClippingPlanes) << '\n';
    s << std::to_string(numClipIntersection) << '\n';

    s << std::to_string(dithering) << '\n';

    s << std::to_string(shadowMapEnabled) << '\n';
    s << std::to_string(as_integer(shadowMapType)) << '\n';

    s << std::to_string(as_integer(toneMapping)) << '\n';
    s << std::to_string(useLegacyLights) << '\n';

    s << std::to_string(premultipliedAlpha) << '\n';

    s << std::to_string(alphaTest) << '\n';
    s << std::to_string(doubleSided) << '\n';
    s << std::to_string(flipSided) << '\n';

    s << std::to_string(depthPacking) << '\n';

    return s.str();
}
