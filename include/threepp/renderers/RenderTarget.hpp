// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderTarget.js

#ifndef THREEPP_RENDERTARGET_HPP
#define THREEPP_RENDERTARGET_HPP

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/textures/Texture.hpp"

#include "threepp/math/Vector4.hpp"
#include "threepp/textures/DepthTexture.hpp"

#include <optional>

namespace threepp {

    class RenderTarget: public EventDispatcher {

    public:
        struct Options {

            std::optional<Mapping> mapping;
            std::optional<TextureWrapping> wrapS;
            std::optional<TextureWrapping> wrapT;
            std::optional<Filter> magFilter;
            std::optional<Filter> minFilter;
            std::optional<Format> format;
            std::optional<Type> type;
            std::optional<int> anisotropy;
            std::optional<ColorSpace> encoding;

            bool generateMipmaps{false};
            bool depthBuffer{true};
            bool stencilBuffer{false};

            // MSAA sample count. 0 (the default) is an ordinary single-sampled
            // target. Above 0, the backend draws into a multisampled
            // renderbuffer pair and resolves into `texture` when the target is
            // unbound, so sampling the texture is unchanged — the extra samples
            // exist only for the duration of the pass. Clamped to the driver's
            // GL_MAX_SAMPLES.
            unsigned int samples{0};

            std::shared_ptr<DepthTexture> depthTexture;

            Options() = default;
        };

        const std::string uuid;

        unsigned int width;
        unsigned int height;
        unsigned int depth = 1;

        Vector4 scissor;
        bool scissorTest = false;

        Vector4 viewport;

        std::shared_ptr<Texture> texture;

        bool depthBuffer;
        bool stencilBuffer;

        // See Options::samples. Changing this after the target has been used
        // has no effect until the target is disposed and re-created.
        unsigned int samples;

        std::shared_ptr<DepthTexture> depthTexture;

        // A target that IS the final picture rather than an intermediate: the
        // GL renderer tone-maps into it inline, exactly as it does when drawing
        // to the screen (three.js does the same for its XR render targets).
        // Every other target receives scene-linear, un-tone-mapped HDR since
        // three.js r154; tone mapping is then applied by whatever displays it
        // (OutputPass at the end of an EffectComposer chain, the Reflector and
        // Water shaders). CameraSensor sets this so a capture matches the
        // on-screen image. Pair it with an sRGB `encoding` for a display-ready
        // 8-bit image.
        bool displayTarget = false;

        RenderTarget(unsigned int width, unsigned int height, const Options& options);

        RenderTarget(RenderTarget&&) = delete;
        RenderTarget(const RenderTarget&) = delete;
        RenderTarget& operator=(RenderTarget&&) = delete;
        RenderTarget& operator=(const RenderTarget&) = delete;

        void setSize(unsigned int width, unsigned int height, unsigned int depth = 1);

        RenderTarget& copy(const RenderTarget& source);

        void dispose();

        static std::unique_ptr<RenderTarget> create(unsigned int width, unsigned int height, const Options& options);

        ~RenderTarget() override;
    };

    // Backward-compatible alias
    using GLRenderTarget = RenderTarget;

}// namespace threepp

#endif//THREEPP_RENDERTARGET_HPP
