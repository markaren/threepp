// Backend-neutral renderer interface.
// GLRenderer derives from this; future Dawn/Vulkan renderers will too.

#ifndef THREEPP_RENDERER_HPP
#define THREEPP_RENDERER_HPP

#include "threepp/constants.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/misc.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace threepp {

    class Camera;
    class Scene;
    class Object3D;
    class RenderTarget;
    class Texture;

    class Renderer {

    public:
        // --- Common configuration (non-virtual, shared by all backends) ---

        bool autoClear = true;
        bool autoClearColor = true;
        bool autoClearDepth = true;
        bool autoClearStencil = true;

        bool sortObjects = true;

        bool shadowMapAutoUpdate = true;

        std::vector<Plane> clippingPlanes;
        bool localClippingEnabled = false;

        float gammaFactor = 2.0f;
        Encoding outputEncoding{Encoding::Linear};

        bool physicallyCorrectLights = false;

        ToneMapping toneMapping{ToneMapping::None};
        float toneMappingExposure = 1.0f;

        bool checkShaderErrors = false;

        // --- Core rendering ---

        virtual void render(Object3D& scene, Camera& camera) = 0;

        // --- Size and pixel ratio ---

        [[nodiscard]] virtual WindowSize size() const = 0;
        virtual void setSize(const std::pair<int, int>& size) = 0;
        [[nodiscard]] virtual float getTargetPixelRatio() const = 0;
        virtual void setPixelRatio(float value) = 0;

        // --- Viewport ---

        virtual void setViewport(const Vector4& v) = 0;
        virtual void setViewport(int x, int y, int width, int height) = 0;

        // --- Scissor ---

        virtual void setScissor(const Vector4& v) = 0;
        virtual void setScissor(int x, int y, int width, int height) = 0;
        virtual void setScissorTest(bool boolean) = 0;

        // --- Shadow map ---

        virtual ShadowMapConfig& shadowMap() { return shadowMapConfig_; }
        [[nodiscard]] virtual const ShadowMapConfig& shadowMap() const { return shadowMapConfig_; }

        // --- Clearing ---

        virtual void setClearColor(const Color& color, float alpha = 1) = 0;
        virtual void getClearColor(Color& target) const {}
        virtual void setClearAlpha(float alpha) {}
        [[nodiscard]] virtual float getClearAlpha() const { return 1.f; }
        virtual void clear(bool color = true, bool depth = true, bool stencil = true) = 0;
        virtual void clearColor() { clear(true, false, false); }
        virtual void clearDepth() { clear(false, true, false); }
        virtual void clearStencil() { clear(false, false, true); }

        // --- Render target ---

        virtual RenderTarget* getRenderTarget() = 0;
        virtual void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) = 0;

        // --- Readback ---

        [[nodiscard]] virtual std::vector<unsigned char> readRGBPixels() = 0;

        virtual void copyTextureToImage(Texture& /*texture*/) {}

        // --- Depth state ---

        virtual void setDepthMask(bool /*flag*/) {}

        // --- Lifecycle ---

        virtual void dispose() = 0;

        virtual ~Renderer() = default;

    protected:
        ShadowMapConfig shadowMapConfig_;
    };

}// namespace threepp

#endif//THREEPP_RENDERER_HPP
