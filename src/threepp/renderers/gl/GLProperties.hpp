// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProperties.js

#ifndef THREEPP_GLPROPERTIES_HPP
#define THREEPP_GLPROPERTIES_HPP

#include "threepp/constants.hpp"
#include "threepp/scenes/Scene.hpp"

#include "GLUniforms.hpp"
#include "threepp/core/Uniform.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/textures/Texture.hpp"

#include <array>
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
        std::optional<std::array<unsigned int, 6>> glCubeFramebuffers;
        std::optional<unsigned int> glDepthbuffer;

        // Multisampled targets (RenderTarget::samples > 0) own a second
        // framebuffer whose attachments are multisampled renderbuffers. Draws
        // go there; glFramebuffer (the one holding the texture) only ever
        // receives the resolve blit.
        std::optional<unsigned int> glMultisampledFramebuffer;
        std::optional<unsigned int> glColorRenderbuffer;
        std::optional<unsigned int> glDepthRenderbuffer;
    };

    struct MaterialProperties {

        GLProgram* currentProgram = nullptr;
        std::unordered_map<std::string, GLProgram*> programs{};

        std::optional<FogVariant> fog;

        int numIntersection{};
        std::optional<int> numClippingPlanes;
        std::vector<float> clippingState;

        Texture* envMap;
        Texture* environment;

        std::optional<ColorSpace> outputEncoding;
        bool instancing{};
        bool skinning{};
        bool vertexAlphas{};

        bool needsLights{};
        bool receiveShadow{};

        // The shadow configuration the current program was compiled against.
        // USE_SHADOWMAP and SHADOWMAP_TYPE_* are baked into the shader, so a
        // runtime change to Renderer::shadowMap() has to invalidate the program —
        // otherwise the shadow pass stops running while the material keeps
        // sampling the (now frozen) shadow map.
        bool shadowMapEnabled{};
        ShadowMap shadowMapType{};

        // Likewise TONE_MAPPING (and the operator's function) and
        // USE_LEGACY_LIGHTS: changing Renderer::toneMapping or useLegacyLights
        // after the material was drawn has to rebuild its program.
        ToneMapping toneMapping{ToneMapping::None};
        bool useLegacyLights{};

        unsigned int lightsStateVersion{};

        std::vector<UniformObject*> uniformsList;
        UniformMap* uniforms;

        // The programs this material has been drawn with, one per variant the
        // renderer switches between while the material itself is unchanged:
        // the side (a transparent double-sided material draws every object
        // twice, back faces then front, and FLIP_SIDED / DOUBLE_SIDED are
        // defines) and the output target (the colour space and the tone
        // mapping are compiled in, so a frame that draws into a render target
        // and then to the screen alternates between two programs for every
        // material). Each entry keeps the uniforms list built for its program,
        // so a switch is a swap. A resolve through getProgram builds the
        // parameters (with copies of both shader sources), the cache key and
        // the list, about 30 us: paid twice an object it made a 70-mesh HUD
        // cost 9 ms a frame, paid once a material a pass it cost about 12 ms a
        // pass on a town-sized scene. Every other parameter is baked into each
        // entry, so the map is cleared whenever the program changes for any
        // other reason.
        struct Variant {
            GLProgram* program = nullptr;
            std::vector<UniformObject*> uniformsList;
        };
        std::unordered_map<unsigned int, Variant> variants;
        std::optional<Side> currentSide;

        static unsigned int variantKey(Side side, ColorSpace encoding, ToneMapping toneMapping) {
            return (static_cast<unsigned int>(side) * 64u + static_cast<unsigned int>(encoding)) * 64u +
                   static_cast<unsigned int>(toneMapping);
        }

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
        GLTypeProperties<RenderTarget, RenderTargetProperties> renderTargetProperties;

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
