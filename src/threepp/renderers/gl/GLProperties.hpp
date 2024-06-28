// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProperties.js

#ifndef THREEPP_GLPROPERTIES_HPP
#define THREEPP_GLPROPERTIES_HPP

#include "threepp/scenes/Scene.hpp"

#include "GLUniforms.hpp"
#include "threepp/core/Uniform.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/renderers/GLRenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include <optional>
#include <unordered_map>

namespace threepp::gl {

    struct GLProgram;

    struct TextureProperties {

        bool glInit{};
        std::optional<int> maxMipLevel{};
        std::optional<unsigned int> glTexture{};
        std::optional<int> currentAnisotropy{};
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

    template<class E, class T>
    struct GLTypeProperties {

        T* get(E* key) {

            return &properties_[key];
        }

        void remove(E* key) {

            properties_.erase(key);
        }

        void dispose() {

            properties_.clear();
        }

    private:
        friend class GLProperties;
        std::unordered_map<E*, T> properties_;
    };

    class GLProperties {

    public:
        GLTypeProperties<Texture, TextureProperties> textureProperties;
        GLTypeProperties<Material, MaterialProperties> materialProperties;
        GLTypeProperties<GLRenderTarget, RenderTargetProperties> renderTargetProperties;

        void dispose() {

            auto texturePropertiesCopy = textureProperties.properties_;
            for (auto& [tex, _] : texturePropertiesCopy) {
                tex->dispose();
            }
            auto materialPropertiesCopy = materialProperties.properties_;
            for (auto& [mat, _] : materialPropertiesCopy) {
                mat->dispose();
            }

            auto renderTargetPropertiesCopy = renderTargetProperties.properties_;
            for (auto& [target, _] : renderTargetPropertiesCopy) {
                target->dispose();
            }

            textureProperties.dispose();
            materialProperties.dispose();
            renderTargetProperties.dispose();
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLPROPERTIES_HPP
