// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLPrograms.js

#ifndef THREEPP_GLPROGRAMS_HPP
#define THREEPP_GLPROGRAMS_HPP

#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/textures/Texture.hpp"

#include <glad/glad.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::gl {

    struct GLPrograms {

        struct Parameters {

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
                    const GLPrograms &scope,
                    Material *material,
                    std::vector<Object3D *> &shadows,
                    std::optional<Fog> fog,
                    int nClipPlanes,
                    int nClipIntersection,
                    Object3D *object);
        };


        bool logarithmicDepthBuffer;
        bool floatVertexTextures;
        GLint maxVertexUniforms;
        bool vertexTextures;

        GLPrograms();

        int getTextureEncodingFromMap(std::optional<Texture> &map) const;
    };

}// namespace threepp::gl

#endif//THREEPP_GLPROGRAMS_HPP
