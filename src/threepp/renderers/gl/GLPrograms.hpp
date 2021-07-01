// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLPrograms.js

#ifndef THREEPP_GLPROGRAMS_HPP
#define THREEPP_GLPROGRAMS_HPP

#include "GLCapabilities.hpp"
#include "GLLights.hpp"
#include "GLProgram.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <threepp/core/Uniform.hpp>
#include <unordered_map>
#include <vector>

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLPrograms {

            struct Parameters;

            std::vector<std::shared_ptr<GLProgram>> programs;

            bool logarithmicDepthBuffer;
            bool floatVertexTextures;
            GLint maxVertexUniforms;
            bool vertexTextures;

            GLPrograms();

            int getTextureEncodingFromMap(std::optional<Texture> &map) const;

            Parameters getParameters(Material *material, const GLLights::LightState &lights, int numShadows, Scene *scene, Object3D *object);

            std::string getProgramCacheKey(const GLRenderer &renderer, const Parameters &parameters);

            std::unordered_map<std::string, Uniform> getUniforms(Material *material);

            std::shared_ptr<GLProgram> acquireProgram(const Parameters &parameters, const std::string &cacheKey);

            void releaseProgram(std::shared_ptr<GLProgram> &program);

            struct Parameters {

                std::optional<std::string> shaderID;
                std::string shaderName;

                std::optional<std::map<std::string, std::string>> defines;

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

                bool alphaTest;
                bool doubleSided;
                bool flipSided;

                bool depthPacking;

                std::string index0AttributeName;

                std::string customProgramCacheKey;

                std::unordered_map<std::string, Uniform> uniforms;

                Parameters(
                        const GLPrograms &scope,
                        Material *material,
                        const GLLights::LightState &lights,
                        int numShadows,
                        Scene *scene,
                        Object3D *object);

                [[nodiscard]] std::string hash() const;

            };
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLPROGRAMS_HPP
