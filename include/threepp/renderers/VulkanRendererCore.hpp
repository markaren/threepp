// Shared base for the two Vulkan renderers.
//
// threepp ships two Vulkan rendering strategies that share almost all of their
// infrastructure — device/swapchain context, acceleration structures, scene
// build + visibility, material/geometry buffers, the raster G-buffer prepass,
// and the bloom / TAA / overlay post-stack. They differ only in how the scene
// is shaded:
//
//   VulkanRenderer    — deferred: a clean analytic G-buffer shade (direct
//                       lights + IBL) plus ray-queried AO/GI. Noise-free,
//                       interactive; the default for synthetic-perception work.
//   VulkanPathTracer  — the reference path tracer: photon caustics + RT
//                       megakernel + à-trous denoiser + ReSTIR DI/GI. The
//                       ground-truth / photorealism path.
//
// This base owns (through the derived leaf's pImpl) all the shared machinery and
// implements the shared public API once. The mode-specific shade is injected
// through the virtual recordSceneDispatch hook (see VulkanRenderer.cpp).

#ifndef THREEPP_VULKANRENDERERCORE_HPP
#define THREEPP_VULKANRENDERERCORE_HPP

#include "threepp/helpers/LidarTypes.hpp"
#include "threepp/renderers/Renderer.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <utility>
#include <vector>

namespace threepp {

    class Mesh;

    class VulkanRendererCore : public Renderer {

    public:
        ~VulkanRendererCore() override;

        VulkanRendererCore(const VulkanRendererCore&) = delete;
        VulkanRendererCore& operator=(const VulkanRendererCore&) = delete;

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        // The actual surface (swapchain) size in PIXELS (differs from size() when
        // OS display scaling is not 100%). Use it for pixel-space math on
        // read-back frames.
        [[nodiscard]] WindowSize framebufferSize() const;

        [[nodiscard]] float getTargetPixelRatio() const override;
        void setPixelRatio(float value) override;

        void setViewport(const Vector4& v) override;
        void setViewport(int x, int y, int width, int height) override;

        void setScissor(const Vector4& v) override;
        void setScissor(int x, int y, int width, int height) override;
        void setScissorTest(bool boolean) override;

        void setClearColor(const Color& color, float alpha = 1) override;
        void getClearColor(Color& target) const override;
        [[nodiscard]] float getClearAlpha() const override;
        void setClearAlpha(float alpha) override;

        void clear(bool color = true, bool depth = true, bool stencil = true) override;

        RenderTarget* getRenderTarget() override;
        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;

        [[nodiscard]] std::vector<unsigned char> readRGBPixels() override;

        // Save the last presented frame to disk (.png/.jpg/.jpeg/.bmp), creating
        // parent dirs as needed. Wraps readRGBPixels(); call after render().
        void writeFramebuffer(const std::filesystem::path& filename) override;

        // Scene-only swapchain capture (post-TAA / pre-overlay) into a host-visible
        // staging buffer. readSceneRGBPixels() returns it without sprite/ImGui
        // overlays — what sensor pipelines need to avoid feedback-looping.
        void setSceneCaptureEnabled(bool enabled);
        [[nodiscard]] bool sceneCaptureEnabled() const;
        [[nodiscard]] std::vector<unsigned char> readSceneRGBPixels();

        // ── GPU event camera (DVS) detector ───────────────────────────────
        struct EventCameraParams {
            float    threshold        = 0.15f;
            float    decay            = 0.85f;
            float    minLuma          = 0.005f;
            uint32_t maxEventsPerPixel = 5;
            uint32_t frameTimeUs       = 0u;
        };

        // 16-byte event record. Matches the GPU layout byte-for-byte.
        struct Event {
            uint32_t x;
            uint32_t y;
            int32_t  polarity;// +1 (bright) / -1 (dark)
            uint32_t t_us;
        };
        void setEventCameraEnabled(bool enabled);
        [[nodiscard]] bool eventCameraEnabled() const;
        void setEventCameraParams(const EventCameraParams& params);
        [[nodiscard]] EventCameraParams eventCameraParams() const;
        [[nodiscard]] std::vector<unsigned char> readEventCameraVisualisation() const;
        size_t readEventCameraVisualisationInto(unsigned char* dst, size_t cap) const;
        void setEventCameraResolution(uint32_t width, uint32_t height);
        [[nodiscard]] std::pair<uint32_t, uint32_t> eventCameraResolution() const;
        size_t readEventStreamInto(Event* dst, size_t cap, bool* overflowed = nullptr) const;
        void setEventsOnlyMode(bool enabled);
        [[nodiscard]] bool eventsOnlyMode() const;

        void dispose() override;

        // ImGui integration handles (Vulkan types erased to void* / uint32_t).
        [[nodiscard]] void* nativeInstance() const;
        [[nodiscard]] void* nativePhysicalDevice() const;
        [[nodiscard]] void* nativeDevice() const;
        [[nodiscard]] void* nativeGraphicsQueue() const;
        [[nodiscard]] uint32_t graphicsQueueFamily() const;
        [[nodiscard]] uint32_t nativeSwapchainFormat() const;// cast to VkFormat
        [[nodiscard]] uint32_t imageCount() const;            // swapchain image count

        // Callback invoked once per frame inside the present render pass, after
        // the scene has been written. The argument is a VkCommandBuffer
        // (type-erased). Set null to disable.
        void setOverlayCallback(std::function<void(void*)> callback);

        // Henyey-Greenstein anisotropy for the fog phase function. Clamped to
        // [-0.95, 0.95]. No effect when scene.fog is unset.
        void setFogAnisotropy(float g);
        [[nodiscard]] float getFogAnisotropy() const;

        // World-Y of the water surface, bounding underwater fog to the water
        // column. Default 1e30 (no limit).
        void setFogWaterSurfaceY(float y);

        // Render scale. The scene shade + hybrid raster G-buffer run at (swapchain
        // extent × scale); TAA reconstructs full resolution. Clamped to
        // [0.25, 1.0]. Default 1.0. Issues a vkDeviceWaitIdle internally — not
        // callable from inside render().
        void setRenderScale(float scale);
        [[nodiscard]] float renderScale() const;

        // Denoiser toggle for the active renderer — PT à-trous or deferred SVGF.
        // Default on.
        void setDenoise(bool enabled);
        [[nodiscard]] bool denoise() const;

        // Deprecated alias of setDenoise()/denoise().
        void setDeferredDenoise(bool enabled);
        [[nodiscard]] bool deferredDenoise() const;

        // Per-NEE-sample firefly clamp used by both the deferred highlight suppress
        // and the PT megakernel. Default 30.0; 0 disables (1e30 sentinel).
        void setFireflyClamp(float cap);
        [[nodiscard]] float fireflyClamp() const;

        // ReSTIR DI master toggle (streaming RIS + temporal + spatial reuse at
        // primary surfaces). Active in the PT megakernel and as a deferred NEE
        // optimization. Off (default) falls back to per-light NEE.
        void setRestirDIEnabled(bool enabled);
        [[nodiscard]] bool restirDIEnabled() const;

        // HDR bloom, added in linear HDR before the tone-map curve. 0 disables.
        void setBloomIntensity(float intensity);
        [[nodiscard]] float bloomIntensity() const;

        // Bloom bright-pass cutoff in linear-HDR luma.
        void setBloomThreshold(float threshold);
        [[nodiscard]] float bloomThreshold() const;

        // Bloom input clamp in linear-HDR luma. <= 0 disables.
        void setBloomClamp(float clampMax);
        [[nodiscard]] float bloomClamp() const;

        // Post-TAA RCAS sharpen strength. 0 disables.
        void setSharpenStrength(float amount);
        [[nodiscard]] float sharpenStrength() const;

        // ── PhysX soft-body zero-copy interop (CUDA → Vulkan) ────────────────
        struct SoftBodyInteropHandle {
            void*  osHandle  = nullptr;
            size_t sizeBytes = 0;
        };
        SoftBodyInteropHandle enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy);
        void disableSoftBodyInterop(const Mesh& mesh);

        // Hybrid-mode raster overlay: post-TAA wireframe / Line / layer-tagged
        // meshes drawn over the shaded image, depth-tested against the raster
        // G-buffer. -1 (default) disables layer selection.
        void setOverlayLayer(int channel);
        [[nodiscard]] int overlayLayer() const;

        // Day-1 / debug visualization: blit one G-buffer channel to the swapchain.
        //   0 = off, 1 = normal, 2 = motion, 3 = instance id, 4 = albedo
        void setHybridDebugView(int view);
        [[nodiscard]] int hybridDebugView() const;

        // ── Path-traced LIDAR scanner ─────────────────────────────────────
        // Synchronously trace beams against the same TLAS, evaluate a back-scatter
        // LIDAR equation at the first hit, and return per-beam tuples. Submits its
        // own command buffer + fence and blocks until results come back.
        void scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& results,
                       const LidarParams& params = {});

        // Per-frame timings (milliseconds). See FrameTimings.
        struct FrameTimings {
            float photonEmitMs   = 0.f;// caustic photon trace (when visible)
            float pathTraceMs    = 0.f;// main RT megakernel / deferred shade
            float denoiseMs      = 0.f;// à-trous passes + finalize tonemap
            float taaMs          = 0.f;// hybrid TAA resolve compute
            float rasterGbufMs   = 0.f;// hybrid G-buffer prepass
            float overlayMs      = 0.f;// hybrid overlay depth + draw
            float cpuEnsureSceneMs = 0.f;// ensureSceneBuilt
            float cpuRecordMs      = 0.f;// recordCommandBuffer
            float cpuFrameMs       = 0.f;// total render() wall time
        };
        [[nodiscard]] FrameTimings lastFrameTimings() const;

    protected:
        VulkanRendererCore() = default;

        // The shared implementation struct. Defined in VulkanRenderer.cpp; each
        // leaf's pImpl derives from it. Opaque to callers of this header.
        struct CoreImpl;

        // Hands the shared base a pointer to the leaf's CoreImpl sub-object so
        // shared behaviour can run without knowing the concrete leaf type.
        [[nodiscard]] virtual CoreImpl* coreImpl() const = 0;
        [[nodiscard]] CoreImpl* core() const { return coreImpl(); }

        // dispose() frees the leaf-owned pImpl; the leaf implements this because
        // it owns the unique_ptr.
        virtual void disposeImpl() = 0;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERERCORE_HPP
