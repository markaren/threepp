
#include "threepp/renderers/CrossRenderer.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/FramebufferTexture.hpp"

#include <algorithm>
#include <cstring>
#include <optional>

namespace threepp {

    struct CrossRenderer::Impl {
        // Visible OpenGL renderer driving the user-supplied canvas.  Renders
        // the scene directly into the left half and the WGPU readback texture
        // into the right half.
        GLRenderer glRenderer;

        // Headless WGPU context that owns the right-half scene render.
        std::optional<Canvas> wgpuCanvas;
        std::unique_ptr<WgpuRenderer> wgpuRenderer;

        // Half-width off-screen target the WGPU renderer draws into.
        std::unique_ptr<RenderTarget> wgpuRT;

        // Texture into which the per-frame WGPU readback is copied; sampled
        // by the right-half display quad.
        std::shared_ptr<FramebufferTexture> wgpuTex;

        // Fullscreen quad scene used for the right-half display blit.
        std::shared_ptr<Scene> previewScene;
        std::shared_ptr<OrthographicCamera> previewCam;
        std::shared_ptr<MeshBasicMaterial> previewMat;

        // User-set render target.  When non-null, render() forwards to the
        // WGPU renderer and skips the split-screen display.
        RenderTarget* userRenderTarget = nullptr;
        int userCubeFace = 0;
        int userMipLevel = 0;

        WindowSize size;

        static int halfWidth(int w) { return std::max(1, w / 2); }

        void allocateOffscreen(int halfW, int h) {
            wgpuRT = RenderTarget::create(halfW, h, RenderTarget::Options{});
            wgpuTex = FramebufferTexture::create(halfW, h);
            wgpuTex->format = Format::RGB;
            wgpuTex->magFilter = Filter::Linear;
            wgpuTex->minFilter = Filter::Linear;
            previewMat->map = wgpuTex;
        }

        explicit Impl(Canvas& glCanvas)
            : glRenderer(glCanvas) {

            const auto sz = glCanvas.size();
            size = sz;
            const int w = sz.width();
            const int h = sz.height();
            const int halfW = halfWidth(w);

            // Scissor test is required so the two halves clear independently.
            glRenderer.setScissorTest(true);

            // Headless WGPU canvas sized to one half of the visible window.
            // Inherit AA from the visible canvas so both halves match.
            wgpuCanvas.emplace(Canvas::Parameters()
                                       .size(halfW, h)
                                       .headless(true)
                                       .vsync(false)
                                       .antialiasing(glCanvas.samples()));
            wgpuRenderer = std::make_unique<WgpuRenderer>(*wgpuCanvas);

            previewScene = Scene::create();
            previewCam = OrthographicCamera::create(-1, 1, 1, -1, 0, 1);
            previewMat = MeshBasicMaterial::create();
            auto previewMesh = Mesh::create(PlaneGeometry::create(2, 2), previewMat);
            previewScene->add(previewMesh);

            allocateOffscreen(halfW, h);
        }

        void resize(int w, int h) {
            if (size.width() == w && size.height() == h) return;
            size = {w, h};

            const int halfW = halfWidth(w);
            glRenderer.setSize({w, h});
            wgpuCanvas->setSize({halfW, h});
            wgpuRenderer->setSize({halfW, h});

            allocateOffscreen(halfW, h);
        }

        void syncStateTo(WgpuRenderer& wgpu, const CrossRenderer& outer) const {
            wgpu.outputColorSpace      = outer.outputColorSpace;
            wgpu.useLegacyLights       = outer.useLegacyLights;
            wgpu.toneMapping           = outer.toneMapping;
            wgpu.toneMappingExposure   = outer.toneMappingExposure;
            wgpu.autoClear             = outer.autoClear;
            wgpu.autoClearColor        = outer.autoClearColor;
            wgpu.autoClearDepth        = outer.autoClearDepth;
            wgpu.autoClearStencil      = outer.autoClearStencil;
            wgpu.sortObjects           = outer.sortObjects;
            wgpu.shadowMapAutoUpdate   = outer.shadowMapAutoUpdate;
            wgpu.clippingPlanes        = outer.clippingPlanes;
            wgpu.localClippingEnabled  = outer.localClippingEnabled;
            wgpu.gammaFactor           = outer.gammaFactor;
            wgpu.checkShaderErrors     = outer.checkShaderErrors;
        }

        void syncStateTo(GLRenderer& gl, const CrossRenderer& outer) const {
            gl.outputColorSpace      = outer.outputColorSpace;
            gl.useLegacyLights       = outer.useLegacyLights;
            gl.toneMapping           = outer.toneMapping;
            gl.toneMappingExposure   = outer.toneMappingExposure;
            gl.autoClear             = outer.autoClear;
            gl.autoClearColor        = outer.autoClearColor;
            gl.autoClearDepth        = outer.autoClearDepth;
            gl.autoClearStencil      = outer.autoClearStencil;
            gl.sortObjects           = outer.sortObjects;
            gl.shadowMapAutoUpdate   = outer.shadowMapAutoUpdate;
            gl.clippingPlanes        = outer.clippingPlanes;
            gl.localClippingEnabled  = outer.localClippingEnabled;
            gl.gammaFactor           = outer.gammaFactor;
            gl.checkShaderErrors     = outer.checkShaderErrors;

            // Mirror shadow config so both halves render shadows consistently.
            // Source of truth is wgpuRenderer's shadowMap (what shadowMap()
            // accessor returns); GLShadowMap inherits ShadowMapConfig, so the
            // base-slice assignment carries enabled/autoUpdate/needsUpdate/type.
            const auto& src = wgpuRenderer->shadowMap();
            auto& dst = gl.shadowMap();
            dst.enabled     = src.enabled;
            dst.autoUpdate  = src.autoUpdate;
            dst.needsUpdate = src.needsUpdate;
            dst.type        = src.type;
        }
    };

    CrossRenderer::CrossRenderer(Canvas& canvas)
        : pimpl_(std::make_unique<Impl>(canvas)) {}

    CrossRenderer::~CrossRenderer() = default;

    GLRenderer& CrossRenderer::glRenderer() { return pimpl_->glRenderer; }
    WgpuRenderer& CrossRenderer::wgpuRenderer() { return *pimpl_->wgpuRenderer; }

    void CrossRenderer::render(Object3D& scene, Camera& camera) {
        auto& gl   = pimpl_->glRenderer;
        auto& wgpu = *pimpl_->wgpuRenderer;

        pimpl_->syncStateTo(gl, *this);
        pimpl_->syncStateTo(wgpu, *this);

        // --- User-bound render target: skip split-screen display. ---
        if (pimpl_->userRenderTarget) {
            wgpu.setRenderTarget(pimpl_->userRenderTarget,
                                 pimpl_->userCubeFace,
                                 pimpl_->userMipLevel);
            wgpu.render(scene, camera);
            return;
        }

        const int w = pimpl_->size.width();
        const int h = pimpl_->size.height();
        const int halfW = Impl::halfWidth(w);

        // Auto-fit perspective camera aspect to a single half so each side is
        // not horizontally squished.  Other camera types (orthographic, etc.)
        // are left untouched — the caller owns their projection bounds.
        auto* perspCam = dynamic_cast<PerspectiveCamera*>(&camera);
        const float halfAspect = static_cast<float>(halfW) / static_cast<float>(h);
        float savedAspect = 0.f;
        bool aspectMutated = false;
        if (perspCam && perspCam->aspect != halfAspect) {
            savedAspect = perspCam->aspect;
            perspCam->aspect = halfAspect;
            perspCam->updateProjectionMatrix();
            aspectMutated = true;
        }

        // --- Right half: WGPU off-screen render. ---
        wgpu.setRenderTarget(pimpl_->wgpuRT.get());
        wgpu.render(scene, camera);
        auto pixels = wgpu.readRGBPixels();

        if (!pixels.empty()) {
            // WGPU readback is top-down; GL textures are bottom-up.
            const int rowBytes = halfW * 3;
            std::vector<unsigned char> flipped(pixels.size());
            for (int row = 0; row < h; ++row) {
                std::memcpy(&flipped[row * rowBytes],
                            &pixels[(h - 1 - row) * rowBytes],
                            rowBytes);
            }
            pimpl_->wgpuTex->image().setData(std::move(flipped));
            pimpl_->wgpuTex->needsUpdate();
        }

        // --- Left half: GL renders the scene directly. ---
        gl.setViewport(0, 0, halfW, h);
        gl.setScissor(0, 0, halfW, h);
        gl.render(scene, camera);

        // --- Right half: GL displays the WGPU readback texture. ---
        // Disable tone mapping so we don't double-map (WGPU already did it).
        const auto savedTone = gl.toneMapping;
        gl.toneMapping = ToneMapping::None;
        gl.setViewport(halfW, 0, w - halfW, h);
        gl.setScissor(halfW, 0, w - halfW, h);
        gl.render(*pimpl_->previewScene, *pimpl_->previewCam);
        gl.toneMapping = savedTone;

        // --- Restore full-window viewport/scissor for any caller-driven
        // overlay (ImGui etc.) that runs after render(). ---
        gl.setViewport(0, 0, w, h);
        gl.setScissor(0, 0, w, h);

        // Restore the caller's camera aspect so they observe no side effects.
        if (aspectMutated) {
            perspCam->aspect = savedAspect;
            perspCam->updateProjectionMatrix();
        }
    }

    WindowSize CrossRenderer::size() const {
        return pimpl_->size;
    }

    void CrossRenderer::setSize(const std::pair<int, int>& sz) {
        pimpl_->resize(sz.first, sz.second);
    }

    float CrossRenderer::getTargetPixelRatio() const {
        return pimpl_->wgpuRenderer->getTargetPixelRatio();
    }

    void CrossRenderer::setPixelRatio(float value) {
        pimpl_->wgpuRenderer->setPixelRatio(value);
        pimpl_->glRenderer.setPixelRatio(value);
    }

    void CrossRenderer::setViewport(const Vector4& v) {
        pimpl_->wgpuRenderer->setViewport(v);
    }
    void CrossRenderer::setViewport(int x, int y, int width, int height) {
        pimpl_->wgpuRenderer->setViewport(x, y, width, height);
    }
    void CrossRenderer::setScissor(const Vector4& v) {
        pimpl_->wgpuRenderer->setScissor(v);
    }
    void CrossRenderer::setScissor(int x, int y, int width, int height) {
        pimpl_->wgpuRenderer->setScissor(x, y, width, height);
    }
    void CrossRenderer::setScissorTest(bool boolean) {
        pimpl_->wgpuRenderer->setScissorTest(boolean);
        pimpl_->glRenderer.setScissorTest(boolean);
    }

    ShadowMapConfig& CrossRenderer::shadowMap() {
        return pimpl_->wgpuRenderer->shadowMap();
    }
    const ShadowMapConfig& CrossRenderer::shadowMap() const {
        return pimpl_->wgpuRenderer->shadowMap();
    }

    void CrossRenderer::setClearColor(const Color& color, float alpha) {
        pimpl_->wgpuRenderer->setClearColor(color, alpha);
        pimpl_->glRenderer.setClearColor(color, alpha);
    }
    void CrossRenderer::getClearColor(Color& target) const {
        pimpl_->wgpuRenderer->getClearColor(target);
    }
    void CrossRenderer::setClearAlpha(float alpha) {
        pimpl_->wgpuRenderer->setClearAlpha(alpha);
        pimpl_->glRenderer.setClearAlpha(alpha);
    }
    float CrossRenderer::getClearAlpha() const {
        return pimpl_->wgpuRenderer->getClearAlpha();
    }
    void CrossRenderer::clear(bool color, bool depth, bool stencil) {
        pimpl_->glRenderer.clear(color, depth, stencil);
        pimpl_->wgpuRenderer->clear(color, depth, stencil);
    }

    RenderTarget* CrossRenderer::getRenderTarget() {
        return pimpl_->userRenderTarget;
    }
    void CrossRenderer::setRenderTarget(RenderTarget* renderTarget, int activeCubeFace, int activeMipmapLevel) {
        pimpl_->userRenderTarget = renderTarget;
        pimpl_->userCubeFace = activeCubeFace;
        pimpl_->userMipLevel = activeMipmapLevel;
        pimpl_->wgpuRenderer->setRenderTarget(renderTarget, activeCubeFace, activeMipmapLevel);
    }

    std::vector<unsigned char> CrossRenderer::readRGBPixels() {
        return pimpl_->wgpuRenderer->readRGBPixels();
    }

    // Render-target work (setRenderTarget / render-to-RT / readback) is routed to
    // the WGPU renderer, so texture/framebuffer copies must go there too — the RT
    // textures live in WGPU's caches. Without this, readback-based helpers such as
    // DepthSensor / LidarSensor would copy nothing and then read an empty buffer.
    void CrossRenderer::copyFramebufferToTexture(const Vector2& position, Texture& texture, int level) {
        pimpl_->wgpuRenderer->copyFramebufferToTexture(position, texture, level);
    }
    void CrossRenderer::copyTextureToImage(Texture& texture) {
        pimpl_->wgpuRenderer->copyTextureToImage(texture);
    }

    bool CrossRenderer::renderTargetFlipY() const {
        return pimpl_->wgpuRenderer->renderTargetFlipY();
    }

    void CrossRenderer::dispose() {
        pimpl_->wgpuRenderer->dispose();
        pimpl_->glRenderer.dispose();
    }

}// namespace threepp
