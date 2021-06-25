
#include "threepp/renderers/gl/GLPrograms.hpp"

#include "threepp/utils/InstanceOf.hpp"

using namespace threepp::gl;

namespace {

    std::unordered_map<std::string, std::string> shaderIds{
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

    // clang-format off
    const std::vector<std::string> parameterNames {
            "precision", "isWebGL2", "supportsVertexTextures", "outputEncoding", "instancing", "instancingColor",
            "map", "mapEncoding", "matcap", "matcapEncoding", "envMap", "envMapMode", "envMapEncoding", "envMapCubeUV",
            "lightMap", "lightMapEncoding", "aoMap", "emissiveMap", "emissiveMapEncoding", "bumpMap", "normalMap", "objectSpaceNormalMap", "tangentSpaceNormalMap", "clearcoatMap", "clearcoatRoughnessMap", "clearcoatNormalMap", "displacementMap", "specularMap",
            "roughnessMap", "metalnessMap", "gradientMap",
            "alphaMap", "combine", "vertexColors", "vertexAlphas", "vertexTangents", "vertexUvs", "uvsVertexOnly", "fog", "useFog", "fogExp2",
            "flatShading", "sizeAttenuation", "logarithmicDepthBuffer", "skinning",
            "maxBones", "useVertexTexture", "morphTargets", "morphNormals", "premultipliedAlpha",
            "numDirLights", "numPointLights", "numSpotLights", "numHemiLights", "numRectAreaLights",
            "numDirLightShadows", "numPointLightShadows", "numSpotLightShadows",
            "shadowMapEnabled", "shadowMapType", "toneMapping", "physicallyCorrectLights",
            "alphaTest", "doubleSided", "flipSided", "numClippingPlanes", "numClipIntersection", "depthPacking", "dithering",
    };
    // clang-format on
}// namespace


GLPrograms::Parameters::Parameters(const GLPrograms &scope, threepp::Material *material, std::vector<Object3D *> &shadows, std::optional<Fog> fog, int nClipPlanes, int nClipIntersection, threepp::Object3D *object) {

    shaderId = shaderIds[material->type()];
    shaderName = material->type();


    isRawShaderMaterial = false; //TODO

    supportsVertexTextures = scope.vertexTextures;


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
