
#include "threepp/renderers/gl/GLPrograms.hpp"

using namespace threepp::gl;

namespace {

    const std::unordered_map<std::string, std::string> shaderIds{
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

struct GLPrograms::Parameters {

    std::string shaderId;
    std::string shaderName;

    std::string vertexShader;
    std::string fragmentShader;

    bool isRawShaderMaterial;

    bool supportsVertexTextures;
    int outputEncoding;

    int numDirLights;
    int numPointLights;
    int numSpotLights;
    int numRectAreaLights;
    int numHemiLights;

    int numDirLightShadows;
    int numPointLightShadows;
    int numSpotLightShadows;

    int numClippingPlanes;
    int numClipIntersection;

    bool dithering;

    bool shadowMapEnabled;
    int shadowMapType;

    int toneMapping;
    bool physicallyCorrectLights;

    bool premultipliedAlpha;

    bool alphaTest;
    bool doubleSided;
    bool flipSided;

    bool depthPacking;

    std::string index0AttributeName;

    std::string customProgramCacheKey;

    Parameters(
            std::shared_ptr<Material> material,
            std::vector<std::shared_ptr<Object3D>> shadows,
            std::optional<Fog> fog,
            int nClipPlanes,
            int nClipIntersection,
            std::shared_ptr<Object3D> object) {

    }
};
