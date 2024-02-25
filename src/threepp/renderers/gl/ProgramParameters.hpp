
#ifndef THREEPP_PROGRAMPARAMETERS_HPP
#define THREEPP_PROGRAMPARAMETERS_HPP

#include "GLClipping.hpp"
#include "GLLights.hpp"
#include "threepp/core/Uniform.hpp"

#include <optional>
#include <string>
#include <unordered_map>

namespace threepp {

    class Scene;
    class GLRenderer;

    namespace gl {

        struct ProgramParameters {

            std::optional<std::string> shaderID;
            std::string shaderName;

            std::unordered_map<std::string, std::string> defines;

            std::string vertexShader;
            std::string fragmentShader;

            bool isRawShaderMaterial{};

            std::string precision = "highp";

            bool instancing{};
            bool instancingColor{};

            bool supportsVertexTextures;
            Encoding outputEncoding{};
            bool map{};
            Encoding mapEncoding{};
            bool matcap{};
            Encoding matcapEncoding{};
            bool envMap{};
            int envMapMode{};
            Encoding envMapEncoding{};
            bool envMapCubeUV{};
            bool lightMap{};
            Encoding lightMapEncoding{};
            bool aoMap{};
            bool emissiveMap{};
            Encoding emissiveMapEncoding{};
            bool bumpMap{};
            bool normalMap{};
            bool objectSpaceNormalMap{};
            bool tangentSpaceNormalMap{};
            bool clearcoatMap{};
            bool clearcoatRoughnessMap{};
            bool clearcoatNormalMap{};
            bool displacementMap{};
            bool roughnessMap{};
            bool metalnessMap{};
            bool specularMap{};
            bool alphaMap{};

            bool gradientMap{};

            std::optional<Color> sheen;

            bool transmission{};
            bool transmissionMap{};
            bool thicknessMap{};

            std::optional<CombineOperation> combine;

            bool vertexTangents{};
            bool vertexColors{};
            bool vertexAlphas{};
            bool vertexUvs{};
            bool uvsVertexOnly{};

            bool fog{};
            bool useFog{};
            bool fogExp2{};

            bool flatShading{};

            bool sizeAttenuation{};
            bool logarithmicDepthBuffer{};

            bool skinning{};
            size_t maxBones{};
            bool useVertexTexture{};

            bool morphTargets{};
            bool morphNormals{};

            size_t numDirLights{};
            size_t numPointLights{};
            size_t numSpotLights{};
            size_t numRectAreaLights{};
            size_t numHemiLights{};

            size_t numDirLightShadows{};
            size_t numPointLightShadows{};
            size_t numSpotLightShadows{};

            int numClippingPlanes{};
            int numClipIntersection{};

            bool dithering{};

            bool shadowMapEnabled{};
            ShadowMap shadowMapType{};

            ToneMapping toneMapping{};
            bool physicallyCorrectLights{};

            bool premultipliedAlpha{};

            float alphaTest{};
            bool doubleSided{};
            bool flipSided{};

            int depthPacking{};

            std::optional<std::string> index0AttributeName;

            UniformMap* uniforms = nullptr;

            ProgramParameters(
                    const GLRenderer& renderer,
                    const GLClipping& clipping,
                    const GLLights::LightState& lights,
                    size_t numShadows,
                    Object3D* object,
                    Scene* scene,
                    Material* material,
                    const std::unordered_map<std::string, std::string>& shaderIDs);

            [[nodiscard]] std::string hash() const;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_PROGRAMPARAMETERS_HPP
