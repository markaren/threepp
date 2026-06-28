#include "vulkan/VulkanCoreImpl.hpp"

namespace threepp {


    // ── VulkanPathTracer ──────────────────────────────────────────────────────
    // The reference path tracer. Shares CoreImpl for all infra (TLAS/BLAS, material
    // buffers, raster G-buffer prepass, post-stack). PT-only state (accumImages, RT
    // pipelines, denoiser, ReSTIR reservoirs, photon caustics, env-CDF) is still in
    // CoreImpl and will be moved here in B2/C. The virtual recordSceneDispatch
    // dispatches the full megakernel path-trace + à-trous denoise pass.
    struct VulkanPathTracer::Impl : VulkanRendererCore::CoreImpl {
        using CoreImpl::CoreImpl;

        bool decalsEnabled() const override { return false; }

        void recordSceneDispatch(VkCommandBuffer cb, uint32_t setIdx,
                                 VkExtent2D ext, VkExtent2D ptExt,
                                 uint32_t exposureBits) override {

            // ── Photon emit pass ────────────────────────────────────────────────
            // Two gates: (1) sceneHasGlass_ — no transmissive material means
            // there's no chance of caustics anywhere, and the chit gather is
            // gated on the same flag. (2) glassVisibleThisFrame_ — even when
            // glass exists, skip the 512×512 emit pass on frames where no
            // glass AABB is in the camera frustum. The world-grid photon
            // store keeps the last emit visible to gather for free, so a few
            // frames of "stale" caustics persist after the camera pans away;
            // they're refreshed the moment glass re-enters view.
            //
            // First-frame safety: chit's gather is gated on sceneHasGlass_
            // alone (not visibility), so if a scene starts with glass but
            // glass is not in the initial frustum we'd let gather read
            // device-local photonCountBuf before anyone has written it.
            // Shim a one-shot zero-fill so the gather always sees a valid
            // (all-zero) count buffer.
            if (sceneHasGlass_ && !glassVisibleThisFrame_ && !photon_->isInitialized()) {
                photon_->recordZeroFillCounts(cb);
            }
            if (sceneHasGlass_ && glassVisibleThisFrame_) {
                gpuTimings_->begin(cb, TP_PhotonEmit, currentFrame);
                // exposureBits hoisted above the mode branch.
                const uint32_t motionFlagsPhoton =
                        (motionThisFrame_      ? 1u : 0u) |
                        (cameraMovedThisFrame_ ? 2u : 0u) |
                        (sceneHasGlass_        ? 4u : 0u);
                uint32_t emPowerBits;
                std::memcpy(&emPowerBits, &emissiveTotalPowerThisFrame_, sizeof(emPowerBits));
                uint32_t envSumBitsPhoton;
                std::memcpy(&envSumBitsPhoton, &envCdfTotalSum_, sizeof(envSumBitsPhoton));
                uint32_t fireflyBitsPhoton;
                std::memcpy(&fireflyBitsPhoton, &fireflyClamp_, sizeof(fireflyBitsPhoton));
                uint32_t oceanFineBitsPhoton;
                std::memcpy(&oceanFineBitsPhoton, &oceanFineTileSize, sizeof(oceanFineBitsPhoton));
                vulkan::PhotonCaustics::EmitPushConstants push{};
                push.v[0]  = sampleIndex;
                push.v[1]  = envImage.mipLevels;
                push.v[2]  = static_cast<uint32_t>(toneMapping_);
                push.v[3]  = exposureBits;
                push.v[4]  = motionFlagsPhoton;
                push.v[5]  = emissiveTriCountThisFrame_;
                push.v[6]  = emPowerBits;
                push.v[7]  = samplesPerPixel_;
                push.v[8]  = envCdfWidth_;
                push.v[9]  = envCdfHeight_;
                push.v[10] = envSumBitsPhoton;
                push.v[11] = fireflyBitsPhoton;
                push.v[12] = oceanFineBitsPhoton;
                photon_->recordEmitPass(cb, descriptorSets[setIdx], push);
                gpuTimings_->end(cb, TP_PhotonEmit, currentFrame);
            }
            // ── End photon emit ─────────────────────────────────────────────────

            const RtPipelineVariant& rtv = activeRtVariant();
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtv.pipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rtPipelineLayout, 0, 1,
                                    &descriptorSets[setIdx], 0, nullptr);

            // [0] sampleIndex (raygen), [1] PMREM mip count (closest_hit),
            // [2] ToneMapping enum, [3] exposure as float bits,
            // [4] motionFlags (bit 0 = any motion, bit 1 = camera moved,
            //                  bit 2 = scene has any glass material,
            //                  bit 3 = hybrid mode (gbuffer drives reproject
            //                          + primary jitter is disabled),
            //                  bit 4 = ReSTIR DI enabled (RIS at primary;
            //                          chit falls back to classic NEE when off),
            //                  bit 5 = perSppJitter (hybrid sub-pixel jitter
            //                          is taken per-sample instead of once),
            //                  bit 6 = ReSTIR GI enabled (Stage 1a single-
            //                          sample indirect reservoir at primary;
            //                          off = classic bounce-1 continuation),
            //                  bit 7 = measure primary trace only — raygen
            //                          exits the step + spp loops immediately
            //                          after the step-0 traceRayEXT; debug
            //                          knob for sizing the bounce-0-on-raster
            //                          opportunity, image goes black),
            //                  bit 8 = ReSTIR DI visibility reuse (shadow-test
            //                          the RIS pick before reuse, discard if
            //                          occluded; only set when bit 4 is too),
            // [5] emissiveCount, [6] emissiveTotalPower (float bits).
            // Per-instance moved bits live in the binding 21 SSBO.
            // (exposure / exposureBits hoisted above the mode branch.)
            const uint32_t motionFlags =
                    (motionThisFrame_           ? 1u   : 0u) |
                    (cameraMovedThisFrame_      ? 2u   : 0u) |
                    (sceneHasGlass_             ? 4u   : 0u) |
                    (restirDIEnabled_           ? 16u  : 0u) |
                    (perSppJitterHybrid_        ? 32u  : 0u) |
                    (restirGIEnabled_           ? 64u  : 0u) |
                    (measurePrimaryTraceOnly_   ? 128u : 0u) |
                    ((restirDIEnabled_ && restirDIVisibilityReuse_) ? 256u : 0u);
            uint32_t emPowerBits;
            std::memcpy(&emPowerBits, &emissiveTotalPowerThisFrame_, sizeof(emPowerBits));
            uint32_t envSumBits;
            std::memcpy(&envSumBits, &envCdfTotalSum_, sizeof(envSumBits));
            uint32_t fireflyBits;
            std::memcpy(&fireflyBits, &fireflyClamp_, sizeof(fireflyBits));
            uint32_t oceanFineBits;
            std::memcpy(&oceanFineBits, &oceanFineTileSize, sizeof(oceanFineBits));
            uint32_t oceanFoamBits;
            std::memcpy(&oceanFoamBits, &oceanFoamTileSize, sizeof(oceanFoamBits));
            const uint32_t pc[16] = {
                    sampleIndex,
                    envImage.mipLevels,
                    static_cast<uint32_t>(toneMapping_),
                    exposureBits,
                    motionFlags,
                    emissiveTriCountThisFrame_,
                    emPowerBits,
                    samplesPerPixel_,
                    envCdfWidth_,
                    envCdfHeight_,
                    envSumBits,
                    fireflyBits,
                    oceanFineBits,
                    edgeMsaaExtra_,
                    oceanFoamBits,
                    maxBounces_,
            };
            vkCmdPushConstants(cb, rtPipelineLayout,
                               VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                       VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                       VK_SHADER_STAGE_MISS_BIT_KHR,
                               0, sizeof(pc), pc);

            // (ext / ptExt hoisted above the mode branch.)
            gpuTimings_->begin(cb, TP_PathTrace, currentFrame);
            ctx->rt().cmdTraceRays(cb, &rtv.rgenRegion, &rtv.missRegion, &rtv.hitRegion,
                                   &callRegion, regionRenderExt_.width, regionRenderExt_.height, 1);
            gpuTimings_->end(cb, TP_PathTrace, currentFrame);

            // ── Spatial denoiser: 2-pass à-trous + finalize tonemap + sRGB ──────
            // RT writes accumImage + gbufImage; denoise pipeline reads them.
            // outImage is owned by the finalize pass (raygen.rgen tail was
            // stripped). All ping-pong slots stay GENERAL throughout. The
            // RT_SHADER → COMPUTE_SHADER barrier is recorded inside
            // Denoiser::recordDispatch.
            gpuTimings_->begin(cb, TP_Denoise, currentFrame);
            denoiser_->recordDispatch(cb, descriptorSets[setIdx], regionRenderExt_,
                                      denoiseEnabled_,
                                      static_cast<uint32_t>(toneMapping_),
                                      exposureBits,
                                      envIsBgColor);
            gpuTimings_->end(cb, TP_Denoise, currentFrame);
            // ── End denoise ─────────────────────────────────────────────────────
        }
    };

    VulkanRendererCore::CoreImpl* VulkanPathTracer::coreImpl() const {
        return pimpl_.get();
    }

    void VulkanPathTracer::disposeImpl() { pimpl_.reset(); }

    VulkanPathTracer::VulkanPathTracer(Canvas& canvas) {
        canvas.initWindow(GraphicsAPI::Vulkan);
        pimpl_ = std::make_unique<Impl>(canvas);
        canvas.setFrameEndCallback([this] {
            if (pimpl_) pimpl_->endFrame();
        });
    }

    VulkanPathTracer::~VulkanPathTracer() = default;


    // ── VulkanPathTracer PT-specific setters ─────────────────────────────────

    void VulkanPathTracer::setSamplesPerPixel(int spp) {
        pimpl_->samplesPerPixel_ = std::max(1, spp);
    }

    int VulkanPathTracer::samplesPerPixel() const {
        return pimpl_->samplesPerPixel_;
    }

    void VulkanPathTracer::setSilhouetteMsaaExtra(uint32_t extra) {
        pimpl_->edgeMsaaExtra_ = extra;
    }

    uint32_t VulkanPathTracer::silhouetteMsaaExtra() const {
        return pimpl_->edgeMsaaExtra_;
    }

    void VulkanPathTracer::setMaxBounces(int bounces) {
        const int clamped = std::clamp(bounces, 1, 16);
        if (clamped == pimpl_->maxBounces_) return;
        pimpl_->maxBounces_ = clamped;
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    int VulkanPathTracer::maxBounces() const {
        return pimpl_->maxBounces_;
    }

    void VulkanPathTracer::resetAccumulation() {
        pimpl_->resetAccumulation();
    }

    void VulkanPathTracer::setPerSppJitterHybrid(bool enabled) {
        pimpl_->perSppJitterHybrid_ = enabled;
    }

    bool VulkanPathTracer::perSppJitterHybrid() const {
        return pimpl_->perSppJitterHybrid_;
    }

    void VulkanPathTracer::setRestirDIVisibilityReuse(bool enabled) {
        pimpl_->restirDIVisibilityReuse_ = enabled;
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanPathTracer::restirDIVisibilityReuse() const {
        return pimpl_->restirDIVisibilityReuse_;
    }

    void VulkanPathTracer::setRestirGIEnabled(bool enabled) {
        pimpl_->restirGIEnabled_ = enabled;
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanPathTracer::restirGIEnabled() const {
        return pimpl_->restirGIEnabled_;
    }

    void VulkanPathTracer::setSerEnabled(bool enabled) {
        pimpl_->serEnabled_ = enabled;
    }

    bool VulkanPathTracer::serEnabled() const {
        return pimpl_->serEnabled_;
    }

    void VulkanPathTracer::setMeasurePrimaryTraceOnly(bool enabled) {
        pimpl_->measurePrimaryTraceOnly_ = enabled;
    }

    bool VulkanPathTracer::measurePrimaryTraceOnly() const {
        return pimpl_->measurePrimaryTraceOnly_;
    }

}// namespace threepp
