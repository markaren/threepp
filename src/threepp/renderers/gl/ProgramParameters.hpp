
#ifndef THREEPP_PROGRAMPARAMETERS_HPP
#define THREEPP_PROGRAMPARAMETERS_HPP

#include "threepp/core/Uniform.hpp"

#include <optional>
#include <string>
#include <unordered_map>

namespace threepp::gl {

    struct ProgramParameters {

        std::optional<std::string> shaderID;
        std::string shaderName;

        std::unordered_map<std::string, std::string> defines;

        std::string vertexShader;
        std::string fragmentShader;

        bool isRawShaderMaterial;

        bool instancing;
        bool instancingColor;

        bool supportsVertexTextures;
        int outputEncoding;
        bool map;
        int mapEncoding;
        bool matcap;
        int matcapEncoding;
        bool envMap;
        int envMapMode;
        int envMapEncoding;
        bool envMapCubeUV;
        bool lightMap;
        int lightMapEncoding;
        bool aoMap;
        bool emissiveMap;
        int emissiveMapEncoding;
        bool bumpMap;
        bool normalMap;
        bool objectSpaceNormalMap;
        bool tangentSpaceNormalMap;
        bool clearcoatMap;
        bool clearcoatRoughnessMap;
        bool clearcoatNormalMap;
        bool displacementMap;
        bool roughnessMap;
        bool metalnessMap;
        bool specularMap;
        bool alphaMap;

        bool gradientMap;

        bool sheen;

        bool transmission;
        bool transmissionMap;
        bool thicknessMap;

        int combine;

        bool vertexTangents;
        bool vertexColors;
        bool vertexAlphas;
        bool vertexUvs;
        bool uvsVertexOnly;

        bool fog;
        bool useFog;
        bool fogExp2;

        bool flatShading;

        bool sizeAttenuation;
        bool logarithmicDepthBuffer;

        int numDirLights;
        int numPointLights;
        int numSpotLights;
        int numRectAreaLights = 0;
        int numHemiLights = 0;

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

        float alphaTest;
        bool doubleSided;
        bool flipSided;

        bool depthPacking;

        std::optional<std::string> index0AttributeName;

        std::unordered_map<std::string, Uniform> uniforms;

        [[nodiscard]] std::string hash() const {

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
    };

}// namespace threepp::gl

#endif//THREEPP_PROGRAMPARAMETERS_HPP
