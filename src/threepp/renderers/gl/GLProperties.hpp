// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProperties.js

#ifndef THREEPP_GLPROPERTIES_HPP
#define THREEPP_GLPROPERTIES_HPP

#include "threepp/scenes/Scene.hpp"

#include "GLUniforms.hpp"
#include "threepp/core/Uniform.hpp"

#include <optional>
#include <unordered_map>

namespace threepp::gl {

    struct GLProgram;

    struct TextureProperties {

        bool glInit{};
        std::optional<int> maxMipLevel{};
        std::optional<unsigned int> glTexture{};
        unsigned int version{};
    };

    struct RenderTargetProperties {

        std::optional<unsigned int> glFramebuffer;
        std::optional<unsigned int> glDepthbuffer;
    };

    struct MaterialProperties {

        GLProgram* program = nullptr;
        GLProgram* currentProgram = nullptr;
        std::unordered_map<std::string, GLProgram*> programs{};

        std::optional<FogVariant> fog;

        int numIntersection{};
        std::optional<int> numClippingPlanes;
        std::vector<float> clippingState;

        Texture* envMap;
        Texture* environment;

        std::optional<Encoding> outputEncoding;
        bool instancing{};
        bool skinning{};
        bool vertexAlphas{};

        bool needsLights{};
        bool receiveShadow{};

        unsigned int lightsStateVersion{};

        std::vector<UniformObject*> uniformsList;
        UniformMap* uniforms;

        unsigned int version{};
    };

    template<class T>
    struct GLTypeProperties {

        T* get(const std::string& key) {

            return &properties_[key];
        }

        void remove(const std::string& key) {

            properties_.erase(key);
        }

        void dispose() {

            properties_.clear();
        }

    private:
        std::unordered_map<std::string, T> properties_;
    };

    struct GLProperties {

        GLTypeProperties<TextureProperties> textureProperties;
        GLTypeProperties<MaterialProperties> materialProperties;
        GLTypeProperties<RenderTargetProperties> renderTargetProperties;

        void dispose() {

            textureProperties.dispose();
            materialProperties.dispose();
            renderTargetProperties.dispose();
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLPROPERTIES_HPP
