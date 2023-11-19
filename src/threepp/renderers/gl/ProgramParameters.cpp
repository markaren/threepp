
#include "threepp/renderers/gl/ProgramParameters.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/shaders/ShaderLib.hpp"

#include "threepp/materials/RawShaderMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <sstream>

using namespace threepp;
using namespace threepp::gl;

namespace {

    Encoding getTextureEncodingFromMap(const std::shared_ptr<Texture>& map) {

        return map ? map->encoding : Encoding::Linear;
    }

}// namespace

ProgramParameters::ProgramParameters(
        const GLRenderer& renderer,
        const GLClipping& clipping,
        const GLLights::LightState& lights,
        size_t numShadows,
        Object3D* object,
        Scene* scene,
        Material* material,
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
    auto roughnessMaterial = dynamic_cast<MaterialWithRoughness*>(material);
    auto metallnessMaterial = dynamic_cast<MaterialWithMetalness*>(material);

    std::string vShader, fShader;
    if (shaderIDs.count(material->type())) {

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
    instancingColor = instancedMesh != nullptr && instancedMesh->instanceColor != nullptr;

    supportsVertexTextures = GLCapabilities::instance().vertexTextures;
    outputEncoding = renderer.outputEncoding;

    map = mapMaterial && mapMaterial->map;
    mapEncoding = getTextureEncodingFromMap(map ? mapMaterial->map : nullptr);
    matcap = matcapMaterial && matcapMaterial->matcap;
    matcapEncoding = getTextureEncodingFromMap(matcap ? matcapMaterial->matcap : nullptr);
    envMap = envmapMaterial && envmapMaterial->envMap;
    if (envMap && envmapMaterial->envMap->mapping){
        envMapMode = as_integer(*envmapMaterial->envMap->mapping);
    }
    envMapEncoding = getTextureEncodingFromMap(envMap ? envmapMaterial->envMap : nullptr);
    envMapCubeUV = envMapMode != 0 &&
                   (envmapMaterial->envMap->mapping == Mapping::CubeReflection ||
                    envmapMaterial->envMap->mapping == Mapping::CubeRefraction);
    lightMap = lightmapMaterial && lightmapMaterial->lightMap;
    lightMapEncoding = getTextureEncodingFromMap(lightMap ? lightmapMaterial->lightMap : nullptr);
    aoMap = aomapMaterial && aomapMaterial->aoMap;
    emissiveMap = emissiveMaterial && emissiveMaterial->emissiveMap;
    emissiveMapEncoding = getTextureEncodingFromMap(emissiveMap ? emissiveMaterial->emissiveMap : nullptr);
    bumpMap = bumpmapMaterial && bumpmapMaterial->bumpMap;
    normalMap = normalMaterial && normalMaterial->normalMap;
    objectSpaceNormalMap = normalMaterial && normalMaterial->normalMapType == ObjectSpaceNormalMap;
    tangentSpaceNormalMap = normalMaterial && normalMaterial->normalMapType == TangentSpaceNormalMap;
    clearcoatMap = false;         //TODO
    clearcoatRoughnessMap = false;//TODO
    clearcoatNormalMap = false;   //TODO
    displacementMap = displacementMapMaterial && displacementMapMaterial->displacementMap;
    roughnessMap = roughnessMaterial && roughnessMaterial->roughnessMap;
    metalnessMap = metallnessMaterial && metallnessMaterial->metalnessMap;
    specularMap = specularMapMaterial && specularMapMaterial->specularMap;
    alphaMap = alphaMaterial && alphaMaterial->alphaMap;

    gradientMap = gradientMaterial && gradientMaterial->gradientMap;

    if (sheenMaterial) {
        sheen = sheenMaterial->sheen;
    }

    transmission = false;   //TODO
    transmissionMap = false;//TODO
    thicknessMap = false;   //TODO

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
    maxBones = 64;    // TODO
    useVertexTexture = GLCapabilities::instance().floatVertexTextures;

    if (auto m = material->as<MaterialWithMorphTargets>()) {
        morphTargets = m->morphTargets;
        morphNormals = m->morphNormals;
    }

    numDirLights = lights.directional.size();
    numPointLights = lights.point.size();
    numSpotLights = lights.spot.size();
    numRectAreaLights = 0;
    numHemiLights = lights.hemi.size();

    numDirLightShadows = lights.directionalShadowMap.size();
    numPointLightShadows = lights.pointShadowMap.size();
    numSpotLightShadows = lights.spotShadowMap.size();

    numClippingPlanes = clipping.numPlanes;
    numClipIntersection = clipping.numIntersection;

    dithering = material->dithering;

    shadowMapEnabled = renderer.shadowMap().enabled && numShadows > 0;
    shadowMapType = renderer.shadowMap().type;

    toneMapping = material->toneMapped ? renderer.toneMapping : ToneMapping::None;
    physicallyCorrectLights = renderer.physicallyCorrectLights;

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
    s << std::to_string(physicallyCorrectLights) << '\n';

    s << std::to_string(premultipliedAlpha) << '\n';

    s << std::to_string(alphaTest) << '\n';
    s << std::to_string(doubleSided) << '\n';
    s << std::to_string(flipSided) << '\n';

    s << std::to_string(depthPacking) << '\n';

    return s.str();
}
