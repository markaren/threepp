// PathTracer — a progressive path trace of the current view over the deferred
// renderer's scene state (path_trace.comp). A FEATURE of VulkanRenderer, not a
// renderer: it binds what the deferred shade binds (TLAS, geometry and
// material records, bindless textures, lights, environment, emitter table,
// fog block, ParticleField density volumes, cloud block, the ocean's fine
// cascade, the primary view's camera), runs as one pass of the frame graph
// while a session is active, and sums its samples into an accumulation buffer
// the host reads back: averaged linear floats (read), or through the
// renderer's display transform to 8-bit (readRGB, path_trace_resolve.comp).
// The frame on screen is untouched.
//
// Driven by VulkanRenderer::pathTraceBegin / pathTraceRead / pathTraceStop:
// each render() while active traces params().samplesPerFrame paths per pixel
// over one band of rows. The picture is params().width x height, or the
// primary view's render extent at each dispatch when those are 0 (a change of
// it restarts the accumulation). A band holds at most as many pixels as the
// render extent, so a picture larger than the frame takes several render()
// calls a pass and no dispatch runs longer than a frame-sized one would.
//
// Needs ray query (VulkanContext::rayQuerySupported); without it the object
// holds no pipeline and available() is false.

#ifndef THREEPP_VULKAN_PATH_TRACER_HPP
#define THREEPP_VULKAN_PATH_TRACER_HPP

#include "threepp/renderers/vulkan/RenderGraph.hpp"
#include "threepp/renderers/vulkan/SpirvReflect.hpp"
#include "threepp/renderers/vulkan/VulkanResources.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace threepp::vulkan {

    class VulkanContext;
    struct OverlayGradeGpu;

    class PathTracer {

    public:
        struct Params {
            uint32_t samplesPerFrame = 8; // camera paths per pixel per render()
            uint32_t maxBounces      = 6; // scattering events after the primary hit
            float    fireflyClamp    = 0.f;// per-sample luminance cap, 0 = off
            uint32_t width           = 0;  // the picture's extent, 0 = the render extent
            uint32_t height          = 0;  // (at most 65535 rows)
            float    domePort        = 0.f;// a dome port's radius about the lens (m), 0 = none
        };

        // What a dispatch reads from the frame besides its descriptor set.
        struct FrameInputs {
            uint32_t emissiveCount      = 0;
            float    emissiveTotalPower = 0.f;
            uint32_t envMipCount        = 1;
            // tan of the sun's angular radius, 0 = a delta sun; sunExtracted:
            // the environment's sun disc was extracted into a directional
            // light, so bounce rays read the sun-clamped mip 1.
            float    sunTanHalfAngle    = 0.f;
            bool     sunExtracted       = false;
            float    oceanFineTileSize  = 0.f;// the fine cascade's tile (m), 0 = no ocean
            float    starIntensity      = 0.f;// the procedural star field, 0 = none
            // setDeferredVolumetrics: the lamps' beam medium in clear air and its g.
            float    volDensity         = 0.f;
            float    volAniso           = 0.f;
            // The camera's lens as the frame forms its image: the
            // distortion (threepp::LensModel, 0 = none) with the frame's
            // normalised intrinsics (fx/W, fy/H, cx/W, cy/H, top-left
            // origin) and the overscan the camera block's frustum carries,
            // and the thin lens (aperture radius in m, 0 = a pinhole) focused
            // at focusDistance.
            uint32_t lensModel          = 0;
            float    lensK[4]{0.5f, 0.5f, 0.5f, 0.5f};
            float    lensRadial[4]{};
            float    lensTangential[2]{};
            float    lensOverscan       = 1.f;
            float    lensRadius         = 0.f;
            float    focusDistance      = 10.f;
        };

        // The subset of DeferredShade's set the paths need, plus the primary
        // view's camera block. Call whenever rewriteDeferredDescriptors runs
        // (TLAS rebuild, env swap, buffer reallocation); idempotent.
        struct DescriptorWriteInputs {
            const VkBuffer* cameraUbo   = nullptr;// [framesInFlight], the primary view's
            const VkBuffer* lightsUbo   = nullptr;// [framesInFlight]
            VkImageView     envView     = VK_NULL_HANDLE;// prefiltered PMREM
            VkSampler       envSampler  = VK_NULL_HANDLE;
            VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
            const VkBuffer* materialBuf = nullptr;// [framesInFlight] MaterialDesc[]
            const VkBuffer* geomDescBuf = nullptr;// [framesInFlight] GeometryDesc[]
            const VkDescriptorImageInfo* materialTex = nullptr;// bindless array
            uint32_t        materialTexCount = 0;
            const VkBuffer* emissiveTriBuf = nullptr;// [framesInFlight] EmTri[]
            const VkBuffer* fogUbo      = nullptr;// [framesInFlight] the air medium
            // ParticleField density volumes (bindings 67-69, as the deferred
            // set has them): kMaxDensityFields views each, every slot filled
            // (the caller's 1x1x1 dummies for the unused ones), and the
            // per-frame box/medium block.
            const VkImageView* particleDensity    = nullptr;// integer volumes
            const VkImageView* particleDensityLin = nullptr;// r16f mirrors
            const VkBuffer*    particleDensityUbo = nullptr;// [framesInFlight]
            // The cloud + height-fog block (binding 58, GpuCloudUbo).
            const VkBuffer*    cloudUbo           = nullptr;// [framesInFlight]
            VkDeviceSize       cloudRange         = VK_WHOLE_SIZE;
            // The ocean's FFT fine-cascade height (binding 13, GENERAL): the
            // chop the wave mesh cannot resolve. A dummy without an ocean.
            VkImageView        oceanFineView      = VK_NULL_HANDLE;
            VkSampler          oceanFineSampler   = VK_NULL_HANDLE;
            // The ocean's wake field and its ripples (bindings 78 and 79,
            // GENERAL): 1x1 dummies without one.
            VkImageView        oceanWakeView      = VK_NULL_HANDLE;
            VkSampler          oceanWakeSampler   = VK_NULL_HANDLE;
            VkImageView        oceanRippleView    = VK_NULL_HANDLE;
            VkSampler          oceanRippleSampler = VK_NULL_HANDLE;
        };

        PathTracer(VulkanContext& ctx, uint32_t framesInFlight);
        ~PathTracer();
        PathTracer(const PathTracer&) = delete;
        PathTracer& operator=(const PathTracer&) = delete;

        [[nodiscard]] bool available() const { return pipe_ != VK_NULL_HANDLE; }

        void rewriteDescriptors(const DescriptorWriteInputs& in, int onlyFrame = -1);
        // Rebind just the emissive-triangle buffer for one frame (the per-frame
        // buffer grows) — mirrors DeferredShade::rewriteEmissive.
        void rewriteEmissive(uint32_t frame, VkBuffer emissiveTriBuf);

        // ── Session ─────────────────────────────────────────────────────────
        // Start (or restart) accumulating. The first dispatch after this
        // overwrites the accumulator.
        void begin(const Params& params);
        void stop() { active_ = false; }
        [[nodiscard]] bool          active() const { return active_; }
        [[nodiscard]] const Params& params() const { return params_; }
        // Samples per pixel accumulated over the whole picture so far (0
        // until the first pass is in).
        [[nodiscard]] uint32_t samples() const { return samples_; }
        // The same with the pass in progress counted by the share of its
        // rows traced: rises with every dispatch.
        [[nodiscard]] double progress() const {
            return height_ ? samples_ + double(params_.samplesPerFrame) * rowCursor_ / height_ : samples_;
        }
        [[nodiscard]] uint32_t width() const { return width_; }
        [[nodiscard]] uint32_t height() const { return height_; }

        // ── Frame graph ─────────────────────────────────────────────────────
        // Size the accumulator for the picture (params' extent, or this
        // frame's render extent; a change restarts the session) and bind it
        // in frame `frame`'s set. Call at graph build, before declare().
        void prepare(uint32_t frame, uint32_t renderWidth, uint32_t renderHeight, uint64_t frameSerial);
        void declare(rg::RenderGraph& graph, rg::PassBuilder& pass, uint32_t frame) const;
        // Trace the next band of rows.
        void recordDispatch(VkCommandBuffer cb, uint32_t frame, const FrameInputs& in);

        // ── Readback ────────────────────────────────────────────────────────
        // Both submit on the graphics queue and wait; the caller drains the
        // device first so every dispatch that wrote the accumulator has
        // finished. False with no samples yet.
        // The averaged picture: linear RGB, three floats per pixel, top row
        // first.
        bool read(std::vector<float>& rgb, uint32_t& w, uint32_t& h);
        // The averaged picture through the display transform: the white
        // balance, exposure, the tone-map operator (threepp::ToneMapping's
        // value), the sRGB encode and the colour grade (`grade`, as the
        // primary view's PostComposite applies them), three bytes per pixel,
        // top row first.
        bool readRGB(std::vector<uint8_t>& rgb, uint32_t& w, uint32_t& h,
                     uint32_t toneMapping, float exposure, const OverlayGradeGpu& grade);

    private:
        VulkanContext& ctx_;
        uint32_t       framesInFlight_;

        std::vector<SpirvBinding>    refl_;
        VkDescriptorSetLayout        dsLayout_   = VK_NULL_HANDLE;
        VkPipelineLayout             pipeLayout_ = VK_NULL_HANDLE;
        VkPipeline                   pipe_       = VK_NULL_HANDLE;
        VkDescriptorPool             descPool_   = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> sets_;// [framesInFlight]
        // The density volumes' samplers: nearest for the integer volumes
        // (texelFetch), linear clamp for the r16f mirrors.
        VkSampler nearestSampler_ = VK_NULL_HANDLE;
        VkSampler linearSampler_  = VK_NULL_HANDLE;

        // The resolve (path_trace_resolve.comp): its own set, written per
        // readRGB from the accumulator and a transient output buffer.
        VkDescriptorSetLayout resolveLayout_     = VK_NULL_HANDLE;
        VkPipelineLayout      resolvePipeLayout_ = VK_NULL_HANDLE;
        VkPipeline            resolvePipe_       = VK_NULL_HANDLE;
        VkDescriptorSet       resolveSet_        = VK_NULL_HANDLE;

        Buffer   accum_{};
        uint32_t width_  = 0;
        uint32_t height_ = 0;
        // The render extent the picture is framed against and the band size
        // it allows; the next row to trace.
        uint32_t renderWidth_  = 0;
        uint32_t renderHeight_ = 0;
        uint32_t bandRows_     = 0;
        uint32_t rowCursor_    = 0;
        // Which accumulator each frame's set names (binding 8); 0 = none.
        std::vector<uint64_t> boundAccum_;
        uint64_t              accumSerial_ = 0;
        // Accumulators replaced by a resize, freed once their last frame is
        // out of flight.
        std::vector<std::pair<Buffer, uint64_t>> retired_;

        Params   params_{};
        bool     active_       = false;
        bool     resetPending_ = false;
        uint32_t samples_      = 0;
        uint32_t seed_         = 0;

        void createPipeline();
        void createResolvePipeline();
        void createDescriptorPool();
        void bindAccumulator(uint32_t frame);
        void freeRetired(uint64_t frameSerial, bool all);
        // Record `record(cb)` in a one-off command buffer on the graphics
        // queue and wait for it.
        template<class F>
        void submitAndWait(F&& record, const char* what);
    };

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_PATH_TRACER_HPP
