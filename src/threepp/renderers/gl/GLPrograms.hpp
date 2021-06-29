// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLPrograms.js

#ifndef THREEPP_GLPROGRAMS_HPP
#define THREEPP_GLPROGRAMS_HPP

#include "GLCapabilities.hpp"
#include "GLLights.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    class GLRenderer;

    namespace gl {

        struct GLPrograms {

            struct Parameters {

                std::optional<std::string> shaderID;
                std::string shaderName;

                std::optional<std::map<std::string, std::string>> defines;

                std::string vertexShader;
                std::string fragmentShader;

                bool isRawShaderMaterial;

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

                int numDirLights;
                int numPointLights;
                int numSpotLights;

                int numDirLightShadows;
                int numPointLightShadows;
                int numSpotLightShadows;
                int numRectAreaLights;
                int numHemiLights;

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
                        const GLPrograms &scope,
                        Material *material,
                        GLLights::LightState &lights,
                        int numShadows,
                        Scene *scene,
                        Object3D *object);
            };


            bool logarithmicDepthBuffer;
            bool floatVertexTextures;
            GLint maxVertexUniforms;
            bool vertexTextures;

            GLPrograms();

            int getTextureEncodingFromMap(std::optional<Texture> &map) const;

            Parameters getParameters(Material *material, GLLights::LightState &lights, int numShadows, Scene *scene, Object3D *object);

            std::string getProgramCacheKey(const GLRenderer& renderer, const Parameters &parameters);
        };

    }

}// namespace threepp::gl

#endif//THREEPP_GLPROGRAMS_HPP
