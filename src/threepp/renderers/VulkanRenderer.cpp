// VulkanRenderer — deferred G-buffer renderer (VulkanRenderer::Impl leaf).
//
// Translation-unit layout:
//   vulkan/VulkanCoreImpl.hpp  — CoreImpl struct (shared infra: TLAS/BLAS, scene
//                                fingerprint, raster G-buffer, TAA/bloom, fog, lights,
//                                skinning, ocean/water, LIDAR, env-PMREM, camera UBOs).
//   VulkanRenderer.cpp (this)  — VulkanRenderer::Impl override (DeferredShade +
//                                auto-exposure) + VulkanRendererCore/VulkanRenderer
//                                public method bodies.
//
// VulkanRenderer (deferred): the raster G-buffer supplies primary visibility;
//   deferred_shade.comp lights it analytically and adds ray-query accents
//   (shadows, reflections, AO/GI), denoised (SVGF) + TAA-resolved.

#define VMA_IMPLEMENTATION
#include "vulkan/VulkanCoreImpl.hpp"

// stb_image_write — implementation is already compiled in GLRenderer.cpp.
// Only this TU calls stbi_write_*() (screenshot/--shot capture path).
#include "stb_image_write.h"

namespace threepp {


    // Deferred renderer's pImpl — adds DeferredShade and deferred-specific state
    // atop the shared VulkanRendererCore::CoreImpl infrastructure.
    struct VulkanRenderer::Impl : VulkanRendererCore::CoreImpl {
        using CoreImpl::CoreImpl;

        // ── Auto-exposure state ───────────────────────────────────────────────
        std::unique_ptr<vulkan::AutoExposure> autoExposure_;
        bool   autoExposureEnabled_ = false;
        float  autoExpSpeed_        = 2.0f;
        float  autoExpMinEV_        = -3.0f;
        float  autoExpMaxEV_        =  3.0f;

        // MSAA dispatch B (per-sample shading at complex/edge pixels) master
        // switch. OFF by default: measured on rock_flicker (msaa=4), dispatch
        // B's fallback variant (single largest non-dominant cluster, cheap
        // env-only diffuse — no RT gather/reflections/ReSTIR, to keep cost
        // proportional to edge-pixel count) made the static-camera flicker
        // metric WORSE (mean ~4600-4800 px/frame) than Phase 1 (dominant-
        // sample resolve) alone (~3700-3900), which was itself already below
        // the msaa=1 baseline (~4000-4030). Root cause: dispatch B's cheap
        // shading model disagrees with dispatch A's full model enough that
        // the disagreement is itself a new, comparably-sized noise source at
        // every edge pixel — trading one flicker mechanism for another
        // instead of removing it. The code compiles, is wired end-to-end,
        // and is architecturally sound (see deferred_shade.comp's shadeMode
        // branches) for a future attempt with a closer-matching per-sample
        // shading model; it just isn't a net win yet, so it stays off.
        // ON by default: under the UNJITTERED msaa raster (see
        // uploadRasterCameraUbo) dispatch B no longer regresses temporal
        // stability (static 23 vs 11 px/frame on the rock harness), and with
        // the corrected coverage accounting (dispatch A blends sky-minority
        // coverage itself; B fills the geometry-minority weight it reserves)
        // it supplies the spatial edge AA the dominant-pick resolve alone
        // lacks. Only consulted when gbufMsaaSamples_ > 1.
        bool gbufShadeBEnabled_ = true;

        // Called once after bloom_->createImages(); wires sceneHdr views.
        void onAfterBloomCreateImages() override {
            if (!autoExposure_) return;
            VkImageView views[kFramesInFlight];
            for (uint32_t f = 0; f < kFramesInFlight; ++f)
                views[f] = bloom_->sceneHdrView(f);
            autoExposure_->rewriteDescriptors(views);
        }

        // CPU per-frame tick: lazy-init, then read histogram + advance EMA.
        void onBeginDeferredFrame(uint32_t frame, float dt) override {
            if (!autoExposureEnabled_) return;
            if (!autoExposure_) {
                autoExposure_ = std::make_unique<vulkan::AutoExposure>(*ctx, kFramesInFlight);
                autoExposure_->adaptSpeed = autoExpSpeed_;
                autoExposure_->minEV      = autoExpMinEV_;
                autoExposure_->maxEV      = autoExpMaxEV_;
                VkImageView views[kFramesInFlight];
                for (uint32_t f = 0; f < kFramesInFlight; ++f)
                    views[f] = bloom_->sceneHdrView(f);
                autoExposure_->rewriteDescriptors(views);
            }
            // Physical camera: the EMA adapts an EV COMPENSATION around the
            // EV100-derived exposure instead of an absolute multiplier
            // around 1.0 (the histogram/EMA machinery is reused unchanged).
            autoExposure_->baseExposure = physicalCamera_ ? physicalExposure() : 1.f;
            autoExposure_->tick(frame, dt);
        }

        // Return adapted exposure when auto-exposure is active (composes the
        // physical-camera base inside AutoExposure::exposure()).
        [[nodiscard]] float currentExposure() const override {
            if (autoExposureEnabled_ && autoExposure_)
                return autoExposure_->exposure();
            return CoreImpl::currentExposure();
        }

        bool decalsEnabled() const override { return true; }

        // HDRI sun → analytic light (see CoreImpl::envSunExtractionWanted).
        // ONE-SUN POLICY: Auto extracts (mip clamp) but injects the analytic
        // sun only while the scene carries no visible DirectionalLight — an
        // explicit scene light claims the sun role (the raster stand-in
        // convention); injecting anyway lit and shadowed the scene with TWO
        // suns (the reported double directional shadow).
        VulkanRenderer::EnvSunPolicy envSunPolicy_ = VulkanRenderer::EnvSunPolicy::Auto;
        bool envSunExtractionWanted() const override {
            return envSunPolicy_ != VulkanRenderer::EnvSunPolicy::Off;
        }
        bool envSunDefersToSceneSun() const override {
            return envSunPolicy_ == VulkanRenderer::EnvSunPolicy::Auto;
        }

        void recordSceneDispatch(VkCommandBuffer cb, uint32_t setIdx,
                                 VkExtent2D ext, VkExtent2D ptExt,
                                 uint32_t exposureBits) override {
            // ── VulkanRenderer deferred dispatch ───────────────────────────
            // Shade the raster material G-buffer (direct analytic lights +
            // split-sum specular IBL + approximate diffuse IBL) straight
            // into bloom_->sceneHdr. No path tracing, no denoise — the base
            // is noise-free. The raster G-buffer pass already ran and its
            // render-pass dependency makes it visible to COMPUTE; bloom's
            // leading barrier makes this write visible to the composite.
            // Per-frame BLAS refits (skinned / deformable meshes) are fenced
            // only to the RT pipeline stage by the build barriers above. The
            // deferred pass traverses the same acceleration structures via
            // ray query from COMPUTE, so add an AS-build → compute fence here.
            // No-op for static scenes (no pending AS write this frame).
            // ALSO carries the GI-reproject cross-frame dependency: this
            // frame's deferred shade SAMPLES the OTHER frame-in-flight's
            // indirect image (last frame's accumulated GI history). Make the
            // prev frame's COMPUTE write to it visible to this frame's COMPUTE
            // read (the GPU executes frames sequentially per queue, so this is a
            // cache-visibility barrier, not ordering).
            {
                VkMemoryBarrier2 asbar{};
                asbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                asbar.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                asbar.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                                      VK_ACCESS_2_SHADER_WRITE_BIT;
                asbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                asbar.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                      VK_ACCESS_2_SHADER_READ_BIT;
                VkDependencyInfo asdep{};
                asdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                asdep.memoryBarrierCount = 1;
                asdep.pMemoryBarriers = &asbar;
                vkCmdPipelineBarrier2(cb, &asdep);
            }
            // ── HiZ pyramid (hybrid SSR) ─────────────────────────────────────
            // Rebuild the closest-depth mip chain from this frame's resolved
            // G-buffer depth (visible to COMPUTE via the raster render pass'
            // dependency). record() carries its own leading/trailing barriers,
            // so the shade dispatch below can sample it with nothing added.
            const bool ssrActive = deferredSsr_ && hiz_ && hiz_->valid();
            if (ssrActive) hiz_->record(cb, currentFrame);
            // ── Probe-GI update (opt-in) ─────────────────────────────────────
            // Refresh a round-robin window of world-space irradiance probes
            // BEFORE the shade so this frame's gather taps a current grid.
            // Runs after the AS barrier above (the probe rays traverse the
            // same TLAS/BLAS). The grid UBO is re-uploaded every frame — its
            // `enabled` flag is what the shader-side sampling gates on.
            if (probeGI_) {
                probeGI_->updateGridUbo(currentFrame, probeGIEnabled_);
                if (probeGIEnabled_) {
                    if (probeGridDirty_) {
                        fitProbeGridToScene();
                        probeGridDirty_ = false;
                        // Grid moved → the UBO written above is stale; rewrite.
                        probeGI_->updateGridUbo(currentFrame, true);
                    }
                    probeGI_->recordDispatch(cb, currentFrame,
                                             emissiveTriCountThisFrame_,
                                             emissiveTotalPowerThisFrame_,
                                             /*shadows=*/true, envImage.mipLevels);
                    // Probe SH writes → deferred shade reads (compute→compute).
                    VkMemoryBarrier2 pbar{};
                    pbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    pbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    pbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    pbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    pbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    VkDependencyInfo pdep{};
                    pdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    pdep.memoryBarrierCount = 1;
                    pdep.pMemoryBarriers = &pbar;
                    vkCmdPipelineBarrier2(cb, &pdep);
                }
            }
            // Dispatch A always sees the TRUE sample count: even with
            // dispatch B off it must blend SKY-minority coverage itself
            // (every geometry/sky silhouette — the most visible edges).
            // shadeBActive (flags bit 7) tells it whether to additionally
            // reserve the geometry-minority weight for dispatch B or fold
            // it into the dominant surface.
            const bool shadeBActive = gbufMsaaSamples_ > 1 && gbufShadeBEnabled_;
            // Clustered light culling: per-cell light lists for the shade's
            // analytic split (all point/spot lights, no 8-per-type cap).
            // Barrier: cull's grid writes → shade's reads (compute→compute).
            if (clusterLightCountThisFrame_ > 0) {
                deferredShade_->recordClusterBuild(cb, currentFrame,
                                                   clusterLightCountThisFrame_,
                                                   regionRenderExt_.width, regionRenderExt_.height);
                VkMemoryBarrier2 cbar{};
                cbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                cbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                cbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                cbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                cbar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo cdep{};
                cdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                cdep.memoryBarrierCount = 1;
                cdep.pMemoryBarriers = &cbar;
                vkCmdPipelineBarrier2(cb, &cdep);
            }
            // Froxel volumetrics: inject (RT sun shafts + clustered-light
            // beams, temporal EMA) + integrate (front-to-back LUT), whenever
            // a medium exists this frame. Runs AFTER the cluster barrier
            // (inject reads the cluster grid); the barrier below makes the
            // LUT visible to the shade's trilinear sample.
            const bool froxelsActive = fogEnabledThisFrame_ || deferredVolDensity_ > 0.f
                                     || heightFogEnabled_;// hetero near-field height fog needs the froxel LUT
            if (froxelsActive) {
                deferredShade_->recordFroxels(cb, currentFrame,
                                              regionRenderExt_.width, regionRenderExt_.height,
                                              deferredVolFog_, deferredVolDensity_, deferredVolAniso_,
                                              sampleIndex,
                                              deferredCamDeltaLen_, deferredCamRotAngle_,
                                              clusterLightCountThisFrame_);
                VkMemoryBarrier2 fbar{};
                fbar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                fbar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                fbar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                fbar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                fbar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                     VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo fdep{};
                fdep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                fdep.memoryBarrierCount = 1;
                fdep.pMemoryBarriers = &fbar;
                vkCmdPipelineBarrier2(cb, &fdep);
            }
            // Half-res volumetric cloud march (cloud_march.comp): raymarch the
            // cloud deck + temporally reproject at half res, off the per-pixel
            // shade critical path. Only when clouds are enabled (off = free /
            // image-identical). Reads the resolved G-buffer depth/ids (already
            // COMPUTE-visible via the raster pass dependency); the barrier below
            // makes its cloudColor write visible to the shade's bilinear sample.
            if (cloudsEnabled_) {
                deferredShade_->recordCloudMarch(cb, currentFrame,
                                                 regionRenderExt_.width, regionRenderExt_.height,
                                                 envImage.mipLevels, sampleIndex,
                                                 deferredCamDeltaLen_, deferredCamRotAngle_);
                VkMemoryBarrier2 cldBar{};
                cldBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                cldBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                cldBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                cldBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                cldBar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                VkDependencyInfo cldDep{};
                cldDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                cldDep.memoryBarrierCount = 1;
                cldDep.pMemoryBarriers = &cldBar;
                vkCmdPipelineBarrier2(cb, &cldDep);
            }
            gpuTimings_->begin(cb, TP_DeferredShade, currentFrame);
            deferredShade_->recordDispatch(cb, currentFrame,
                                           regionRenderExt_.width, regionRenderExt_.height,
                                           envImage.mipLevels, /*shadows=*/true,
                                           /*ao=*/deferredAO_, sampleIndex,
                                           emissiveTriCountThisFrame_,
                                           emissiveTotalPowerThisFrame_,
                                           fireflyClamp_,
                                           oceanFineTileSize, oceanFoamTileSize,
                                           denoiseEnabled_, restirDIEnabled_, deferredVolFog_,
                                           deferredVolDensity_, deferredVolAniso_,
                                           deferredStarIntensity_,
                                           deferredCamDeltaLen_, deferredCamRotAngle_,
                                           static_cast<float>(glfwGetTime()),
                                           std::tan(sunAngularRadiusDeg_ * 0.017453292519943295f),
                                           gbufMsaaSamples_, /*shadeMode=*/0u, shadeBActive,
                                           clusterLightCountThisFrame_, froxelsActive,
                                           ssrActive, preExpBits_, prevPreExpBits_,
                                           envIsBgColor);
            gpuTimings_->end(cb, TP_DeferredShade, currentFrame);// pathTraceMs = deferred SHADE only

            // ── MSAA dispatch B: per-sample shading at complex (edge) pixels ──
            // Opt-in (gbufShadeBEnabled_, default false) and only when
            // setGbufferMsaa(2|4) is active. Reads dispatch A's outImage
            // write (imageLoad accumulate) and the raw MS G-buffer; needs a
            // compute->compute barrier on outImage between the two
            // dispatches (RAW: B reads what A just wrote) plus visibility for
            // the MS attachments (already satisfied — they've been
            // SHADER_READ_ONLY since the MSAA render pass's own subpass
            // dependency, unchanged since dispatch A started).
            if (gbufMsaaSamples_ > 1 && gbufShadeBEnabled_) {
                VkMemoryBarrier2 shadeBar{};
                shadeBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                shadeBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                shadeBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                shadeBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                shadeBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo shadeDep{};
                shadeDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                shadeDep.memoryBarrierCount = 1;
                shadeDep.pMemoryBarriers    = &shadeBar;
                vkCmdPipelineBarrier2(cb, &shadeDep);

                gpuTimings_->begin(cb, TP_ShadeB, currentFrame);
                deferredShade_->recordDispatch(cb, currentFrame,
                                               regionRenderExt_.width, regionRenderExt_.height,
                                               envImage.mipLevels, /*shadows=*/true,
                                               /*ao=*/deferredAO_, sampleIndex,
                                               emissiveTriCountThisFrame_,
                                               emissiveTotalPowerThisFrame_,
                                               fireflyClamp_,
                                               oceanFineTileSize, oceanFoamTileSize,
                                               denoiseEnabled_, restirDIEnabled_, deferredVolFog_,
                                               deferredVolDensity_, deferredVolAniso_,
                                               deferredStarIntensity_,
                                               deferredCamDeltaLen_, deferredCamRotAngle_,
                                               static_cast<float>(glfwGetTime()),
                                               std::tan(sunAngularRadiusDeg_ * 0.017453292519943295f),
                                               gbufMsaaSamples_, /*shadeMode=*/1u, /*shadeBActive=*/true,
                                               clusterLightCountThisFrame_, froxelsActive,
                                               ssrActive, preExpBits_, prevPreExpBits_,
                                               envIsBgColor);
                gpuTimings_->end(cb, TP_ShadeB, currentFrame);

                // Dispatch B's outImage write -> bloom/composite's read.
                VkMemoryBarrier2 postBar{};
                postBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                postBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                postBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                postBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                postBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo postDep{};
                postDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                postDep.memoryBarrierCount = 1;
                postDep.pMemoryBarriers    = &postBar;
                vkCmdPipelineBarrier2(cb, &postDep);
            }
            // Spatial denoise of the demodulated diffuse-indirect + recombine.
            // Barrier: the shade wrote sceneHdr + the indirect image (both
            // GENERAL storage); the denoise reads the indirect 5×5 neighbourhood
            // and read-modify-writes sceneHdr — compute→compute RAW/WAR.
            if (denoiseEnabled_) {
                VkMemoryBarrier2 denoiseBar{};
                denoiseBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                denoiseBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                denoiseBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                denoiseBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                denoiseBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                VkDependencyInfo denoiseDep{};
                denoiseDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                denoiseDep.memoryBarrierCount = 1;
                denoiseDep.pMemoryBarriers = &denoiseBar;
                vkCmdPipelineBarrier2(cb, &denoiseDep);
                gpuTimings_->begin(cb, TP_Denoise, currentFrame);// denoiseMs = deferred SVGF (4 GI passes + reflection pass)
                deferredShade_->recordDenoiseDispatch(cb, currentFrame, regionRenderExt_.width, regionRenderExt_.height,
                                                      gbufMsaaSamples_, shadeBActive, preExpBits_);
                gpuTimings_->end(cb, TP_Denoise, currentFrame);
            }
            // Auto-exposure: histogram over the final sceneHdr. sceneHdr writes
            // (deferred shade + optional denoise) are already visible via the
            // barriers above; bloom's leading barrier will also make them visible,
            // so this fits naturally in the gap. recordDispatch() inserts its own
            // fill→compute barrier to zero the SSBO before sampling.
            if (autoExposureEnabled_ && autoExposure_) {
                VkMemoryBarrier2 lumBar{};
                lumBar.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                lumBar.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                lumBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                lumBar.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                lumBar.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                VkDependencyInfo lumDep{};
                lumDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                lumDep.memoryBarrierCount = 1;
                lumDep.pMemoryBarriers    = &lumBar;
                vkCmdPipelineBarrier2(cb, &lumDep);
                autoExposure_->recordDispatch(cb, currentFrame,
                                             regionRenderExt_.width, regionRenderExt_.height,
                                             preExpHist_[currentFrame]);// meter un-bakes this
            }
        }
    };

    VulkanRendererCore::CoreImpl* VulkanRenderer::coreImpl() const {
        return pimpl_.get();
    }

    void VulkanRenderer::disposeImpl() { pimpl_.reset(); }

    VulkanRendererCore::~VulkanRendererCore() = default;

    VulkanRenderer::VulkanRenderer(Canvas& canvas) {
        canvas.initWindow(GraphicsAPI::Vulkan);
        pimpl_ = std::make_unique<Impl>(canvas);
        // Mirror WgpuRenderer: the user's animate lambda may call render()
        // multiple times in one iteration (e.g. main scene + HUD overlay
        // via threepp::HUD). Each render() opens or extends the in-flight
        // frame; the present is deferred to the canvas frame-end callback
        // so all draws land on the same swapchain image. See
        // Impl::endFrame() for the submit+present body.
        canvas.setFrameEndCallback([this] {
            if (pimpl_) pimpl_->endFrame();
        });
    }

    VulkanRenderer::~VulkanRenderer() = default;

    void VulkanRendererCore::render(Object3D& scene, Camera& camera) {
        const auto frameStart = std::chrono::high_resolution_clock::now();
        const auto cur = core()->canvas.size();
        if (cur.width() != core()->size.width() || cur.height() != core()->size.height()) {
            core()->needsResize = true;
        }
        // Mirror Renderer-base tone-mapping state into the Impl so renderFrame
        // can push it as a single 16-byte block. Done every render() so users
        // can flip toneMapping / toneMappingExposure freely between frames.
        core()->toneMapping_         = toneMapping;
        core()->toneMappingExposure_ = toneMappingExposure;
        core()->autoClear_           = autoClear;
        // A split-screen secondary pane — a second perspective render() into a
        // scissor sub-rect while a frame is already in flight — composes
        // overlay-only (Points / Lines / Sprites) into that region and must NOT
        // run the scene-build pass. renderFrame already routes it to the
        // overlay-only path (its matching condition below), but the open frame's
        // command buffer still has the PRIMARY pane's TLAS + scene-desc buffers
        // bound: ensureSceneBuilt's structural-rebuild branch (a different scene
        // ⇒ snapshot mismatch ⇒ fullRebuild) tears those down mid-frame, which
        // invalidates the recording command buffer and surfaces as a device-lost
        // at the next buildTlas one-shot. OverlayPass::record updates the
        // scene/camera matrices itself, so skipping the build here is matrix-safe.
        const bool secondaryOverlayPane =
                core()->frameState_ != CoreImpl::FrameState::Idle &&
                core()->scissorTest &&
                core()->scissor.z >= 1.f && core()->scissor.w >= 1.f;
        // Only the deferred-render-bound (perspective-camera) primary
        // render() call needs the scene-build pass — it populates
        // lastVisibleEntries_, the BLAS cache, motion bits and the per-mesh
        // fingerprint state the deferred shade pipeline reads. The HUD
        // pattern's second call (ortho camera over a separate HUD scene)
        // must not touch any of that, or it clobbers meshMovedBits_ /
        // lastVisibleEntries_ and the next deferred frame
        // cold-starts (visibly drops to ~1-spp quality).
        // The ortho overlay record path walks the HUD scene directly instead.
        if (!camera.is<OrthographicCamera>() && !secondaryOverlayPane) {
            const auto sceneStart = std::chrono::high_resolution_clock::now();
            core()->ensureSceneBuilt(scene, camera);
            // World-space Sprites (screenSpace == false) are drawn by the overlay
            // billboard pass, not the deferred/G-buffer path. Snapshot them each frame
            // with fresh world matrices (ensureSceneBuilt just ran
            // updateMatrixWorld) — independent of the snapshot/lean machinery,
            // since impact sprites move/spawn/expire every frame.
            core()->collectWorldSprites(scene);
            core()->pendingCpuEnsureSceneMs_ =
                    std::chrono::duration<float, std::milli>(
                            std::chrono::high_resolution_clock::now() - sceneStart)
                            .count();
        }
        core()->renderFrame(scene, camera);
        core()->gpuTimings_->setCpuFrameMs(
                std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - frameStart)
                        .count());
    }

    WindowSize VulkanRendererCore::size() const { return core()->size; }

    WindowSize VulkanRendererCore::framebufferSize() const {
        auto* ctx = core()->ctx.get();
        if (!ctx || ctx->swapchainImages().empty()) return core()->size;
        const VkExtent2D ext = ctx->swapchainExtent();
        return {static_cast<int>(ext.width), static_cast<int>(ext.height)};
    }

    void VulkanRendererCore::setSize(const std::pair<int, int>& s) {
        core()->size = WindowSize{s.first, s.second};
        core()->needsResize = true;
    }

    // Pixel ratio is a GL/canvas concept (logical-pixel → device-pixel scale).
    // The Vulkan swapchain is already sized in native device pixels, so there
    // is no logical domain to scale — the ratio is always 1 and the setter is
    // a warned no-op rather than misleading mutable state. Resolution scaling
    // goes through setRenderScale, the one supported lever.
    float VulkanRendererCore::getTargetPixelRatio() const { return 1.f; }
    void VulkanRendererCore::setPixelRatio(float) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[VulkanRenderer] setPixelRatio: unsupported (the swapchain is "
                         "native-pixel; use setRenderScale for resolution scaling) — "
                         "call ignored\n";
        }
    }

    void VulkanRendererCore::setViewport(const Vector4& v) { core()->viewport = v; }
    void VulkanRendererCore::setViewport(int x, int y, int w, int h) {
        core()->viewport.set(static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(w), static_cast<float>(h));
    }

    void VulkanRendererCore::setScissor(const Vector4& v) { core()->scissor = v; }
    void VulkanRendererCore::setScissor(int x, int y, int w, int h) {
        core()->scissor.set(static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(w), static_cast<float>(h));
    }
    void VulkanRendererCore::setScissorTest(bool b) { core()->scissorTest = b; }

    void VulkanRendererCore::setClearColor(const Color& c, float a) {
        core()->clearColor = c;
        core()->clearAlpha = a;
    }
    void VulkanRendererCore::getClearColor(Color& target) const { target = core()->clearColor; }
    float VulkanRendererCore::getClearAlpha() const { return core()->clearAlpha; }
    void VulkanRendererCore::setClearAlpha(float a) { core()->clearAlpha = a; }

    void VulkanRendererCore::clear(bool, bool, bool) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[VulkanRenderer] clear(): unsupported — the deferred pipeline "
                         "rewrites every attachment each render(); call ignored\n";
        }
    }

    // nullptr = "rendering to the default framebuffer" in three.js semantics —
    // accurate here, the swapchain is the only target this renderer has.
    RenderTarget* VulkanRendererCore::getRenderTarget() { return nullptr; }
    void VulkanRendererCore::setRenderTarget(RenderTarget* renderTarget, int, int) {
        if (!renderTarget) return;// null = default framebuffer — already the only mode
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::cerr << "[VulkanRenderer] setRenderTarget(): offscreen render targets are "
                         "unsupported (swapchain-only renderer); call ignored — use "
                         "readGBufferAOV/readRGBPixels for capture\n";
        }
    }

    void VulkanRendererCore::writeFramebuffer(const std::filesystem::path& filename) {
        auto ext = filename.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp") {
            throw std::runtime_error("VulkanRendererCore::writeFramebuffer: unsupported format " + ext);
        }
        const auto pixels = readRGBPixels();
        if (pixels.empty()) {
            throw std::runtime_error("VulkanRendererCore::writeFramebuffer: no readable framebuffer");
        }
        const auto sz = size();
        const int  w  = sz.width();
        const int  h  = sz.height();
        if (filename.has_parent_path() && !std::filesystem::exists(filename.parent_path())) {
            std::error_code ec;
            std::filesystem::create_directories(filename.parent_path(), ec);
        }
        bool success = false;
        if (ext == ".png") {
            success = stbi_write_png(filename.string().c_str(), w, h, 3, pixels.data(), w * 3);
        } else if (ext == ".jpg" || ext == ".jpeg") {
            success = stbi_write_jpg(filename.string().c_str(), w, h, 3, pixels.data(), 100);
        } else {
            success = stbi_write_bmp(filename.string().c_str(), w, h, 3, pixels.data());
        }
        if (!success) {
            throw std::runtime_error("VulkanRenderer: failed to write framebuffer to " + filename.string());
        }
    }

    std::vector<unsigned char> VulkanRendererCore::readRGBPixels() {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx || ctx->swapchainImages().empty()) return {};

        const VkExtent2D ext   = ctx->swapchainExtent();
        const auto       w     = ext.width;
        const auto       h     = ext.height;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
        if (w == 0 || h == 0) return {};

        // The copy below is only legal when the swapchain was created with
        // TRANSFER_SRC usage (see VulkanContext::createSwapchain). Universally
        // available on desktop; fail loudly rather than issue an invalid copy.
        if (!ctx->swapchainSupportsTransferSrc()) {
            throw std::runtime_error(
                    "VulkanRenderer::readRGBPixels: the surface does not support "
                    "TRANSFER_SRC swapchain usage, so the presented image cannot be "
                    "copied out on this platform");
        }

        // Wait so the previously presented swapchain image is fully written
        // and stable. Cheap unless the user is hammering render() — they
        // usually aren't between an interactive render() and a readback.
        vkDeviceWaitIdle(ctx->device());

        // Allocate a host-visible staging buffer. Reuses the same allocator
        // pattern the LIDAR scanner uses for its readback path.
        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        // One-shot transient command buffer. Doesn't share the main
        // per-frame command pool because we need to submit + wait
        // synchronously without disturbing the in-flight frame state.
        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readRGBPixels)");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readRGBPixels)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readRGBPixels)");

        const VkImage src = ctx->swapchainImages()[impl.frameImageIndex_];

        // Transition swapchain image PRESENT_SRC → TRANSFER_SRC for the copy.
        VkImageMemoryBarrier toSrc{};
        toSrc.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout                   = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toSrc.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcAccessMask               = 0;
        toSrc.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image                       = src;
        toSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toSrc.subresourceRange.levelCount = 1;
        toSrc.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {w, h, 1};
        vkCmdCopyImageToBuffer(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.handle, 1, &region);

        // Restore PRESENT_SRC layout so the next frame can present this slot.
        VkImageMemoryBarrier toPresent = toSrc;
        toPresent.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toPresent.newLayout            = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
        toPresent.dstAccessMask        = 0;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toPresent);

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readRGBPixels)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readRGBPixels)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readRGBPixels)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readRGBPixels)");

        // Map staging, convert BGRA8_UNORM → RGB8 (Vulkan picks BGRA in
        // VulkanContext::createSwapchain; the surface format is fixed).
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readRGBPixels)");
        // Invalidate AFTER mapping: vkInvalidateMappedMemoryRanges requires
        // the memory to be currently host mapped (ad-hoc-mapped staging — a
        // persistently-mapped buffer wouldn't care about the ordering).
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, bytes);
        const auto* bgra = static_cast<const unsigned char*>(mapped);
        const size_t pixels = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = bgra[i * 4 + 2];// R ← B-channel of source
            rgb[i * 3 + 1] = bgra[i * 4 + 1];// G ← G
            rgb[i * 3 + 2] = bgra[i * 4 + 0];// B ← R
        }
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);

        return rgb;
    }

    void VulkanRendererCore::setSceneCaptureEnabled(bool enabled) {
        // Scene capture copies the mid-frame swapchain image into a staging
        // buffer (recordSceneCapture) — same TRANSFER_SRC precondition as
        // readRGBPixels. ctx may not exist yet (pre-first-render enable);
        // recordSceneCapture re-checks and skips with a warning in that case.
        if (enabled && core()->ctx && !core()->ctx->swapchainSupportsTransferSrc()) {
            throw std::runtime_error(
                    "VulkanRenderer::setSceneCaptureEnabled: the surface does not "
                    "support TRANSFER_SRC swapchain usage, so scene capture is "
                    "unavailable on this platform");
        }
        core()->sceneCaptureEnabled_ = enabled;
    }

    bool VulkanRendererCore::sceneCaptureEnabled() const {
        return core()->sceneCaptureEnabled_;
    }

    std::vector<unsigned char> VulkanRendererCore::readSceneRGBPixels() {
        auto& impl = *core();
        if (!impl.sceneCaptureEnabled_ || impl.sceneCaptureBuf_.handle == VK_NULL_HANDLE) {
            return {};
        }

        // Wait so the most recent frame's capture is flushed before we
        // memcpy. Same trade-off as readRGBPixels — cheap unless the
        // caller hammers it back-to-back.
        vkDeviceWaitIdle(impl.ctx->device());

        const uint32_t w = impl.sceneCaptureBufW_;
        const uint32_t h = impl.sceneCaptureBufH_;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4;
        if (bytes == 0) return {};

        void* mapped = nullptr;
        if (vmaMapMemory(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, &mapped) != VK_SUCCESS) {
            return {};
        }
        // AFTER mapping — see readRGBPixels for the ordering rationale.
        vulkan::invalidateHostReads(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, 0, bytes);
        const auto* bgra = static_cast<const unsigned char*>(mapped);
        std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3);
        const size_t pixels = static_cast<size_t>(w) * h;
        for (size_t i = 0; i < pixels; ++i) {
            rgb[i * 3 + 0] = bgra[i * 4 + 2];// R ← B
            rgb[i * 3 + 1] = bgra[i * 4 + 1];// G ← G
            rgb[i * 3 + 2] = bgra[i * 4 + 0];// B ← R
        }
        vmaUnmapMemory(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc);
        return rgb;
    }

    bool VulkanRendererCore::readGBufferAOV(GBufferAOV aov, std::vector<uint8_t>& out,
                                            int& width, int& height, int& bytesPerPixel) {
        auto& impl = *core();
        auto* ctx  = impl.ctx.get();
        if (!ctx) return false;

        // The just-rendered G-buffer sits in the slot BEFORE the current one:
        // endFrame advances currentFrame after recording (VulkanCoreImpl.hpp),
        // so the freshest attachment contents are (currentFrame - 1) mod N —
        // the same slot arithmetic the fog history uses.
        const uint32_t n    = static_cast<uint32_t>(impl.rasterGbufs.size());
        const uint32_t slot = (impl.currentFrame + n - 1u) % n;
        const auto& g       = impl.rasterGbufs[slot];

        // Select the attachment, its aspect, and the layout it rests in after a
        // frame (the raster render pass' finalLayout; the MSAA resolve leaves the
        // resolved single-sample images in the same layouts). Depth carries the
        // depth aspect + DEPTH_STENCIL_READ_ONLY; every colour AOV is SHADER_READ_ONLY.
        const vulkan::Image2D* img = nullptr;
        VkImageAspectFlags aspect  = VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageLayout restLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        switch (aov) {
            case GBufferAOV::Depth:
                img = &g.depth; aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                restLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL; break;
            case GBufferAOV::Normal: img = &g.normal; break;
            case GBufferAOV::Motion: img = &g.motion; break;
            case GBufferAOV::Ids:    img = &g.ids;    break;
            case GBufferAOV::Albedo: img = &g.albedo; break;
        }
        if (!img || img->image == VK_NULL_HANDLE || img->width == 0 || img->height == 0) {
            return false;// no frame rendered yet
        }

        const uint32_t w = img->width;
        const uint32_t h = img->height;
        // Element size of the attachment format: RGBA16 (normal/motion/ids) = 8,
        // D32_SFLOAT (depth) and RGBA8_UNORM (albedo) = 4.
        uint32_t bpp = 4;
        if (img->format == VK_FORMAT_R16G16B16A16_SFLOAT ||
            img->format == VK_FORMAT_R16G16B16A16_UINT) {
            bpp = 8;
        }
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * bpp;

        // Wait so the last frame's writes to this attachment are complete and no
        // in-flight frame is still sampling it — same trade-off as readRGBPixels.
        vkDeviceWaitIdle(ctx->device());

        vulkan::Buffer staging = vulkan::createBuffer(
                ctx->allocator(), ctx->device(), bytes,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = ctx->queueFamilies().graphics;
        VkCommandPool cpool = VK_NULL_HANDLE;
        vulkan::check(vkCreateCommandPool(ctx->device(), &cpci, nullptr, &cpool),
                      "vkCreateCommandPool(readGBufferAOV)");

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool        = cpool;
        cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        vulkan::check(vkAllocateCommandBuffers(ctx->device(), &cbai, &cb),
                      "vkAllocateCommandBuffers(readGBufferAOV)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vulkan::check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer(readGBufferAOV)");

        // restLayout → TRANSFER_SRC for the copy. srcAccess 0 is safe: the prior
        // vkDeviceWaitIdle already retired every access, so this barrier only
        // performs the layout transition.
        VkImageMemoryBarrier toSrc{};
        toSrc.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout                   = restLayout;
        toSrc.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.srcAccessMask               = 0;
        toSrc.dstAccessMask               = VK_ACCESS_TRANSFER_READ_BIT;
        toSrc.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        toSrc.image                       = img->image;
        toSrc.subresourceRange.aspectMask = aspect;
        toSrc.subresourceRange.levelCount = 1;
        toSrc.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toSrc);

        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = aspect;
        region.imageSubresource.layerCount = 1;
        region.imageExtent                 = {w, h, 1};
        vkCmdCopyImageToBuffer(cb, img->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.handle, 1, &region);

        // Restore the resting layout so the next frame's consumers (deferred
        // shade / TAA) find the attachment where they expect it.
        VkImageMemoryBarrier toRest = toSrc;
        toRest.oldLayout            = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toRest.newLayout            = restLayout;
        toRest.srcAccessMask        = VK_ACCESS_TRANSFER_READ_BIT;
        toRest.dstAccessMask        = 0;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toRest);

        vulkan::check(vkEndCommandBuffer(cb), "vkEndCommandBuffer(readGBufferAOV)");

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        vulkan::check(vkCreateFence(ctx->device(), &fci, nullptr, &fence),
                      "vkCreateFence(readGBufferAOV)");

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        vulkan::check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, fence),
                      "vkQueueSubmit(readGBufferAOV)");
        vulkan::check(vkWaitForFences(ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX),
                      "vkWaitForFences(readGBufferAOV)");

        out.resize(static_cast<size_t>(bytes));
        void* mapped = nullptr;
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readGBufferAOV)");
        // AFTER mapping — see readRGBPixels for the ordering rationale.
        vulkan::invalidateHostReads(ctx->allocator(), staging.alloc, 0, bytes);
        std::memcpy(out.data(), mapped, static_cast<size_t>(bytes));
        vmaUnmapMemory(ctx->allocator(), staging.alloc);

        vkDestroyFence(ctx->device(), fence, nullptr);
        vkDestroyCommandPool(ctx->device(), cpool, nullptr);
        vulkan::destroyBuffer(ctx->allocator(), staging);

        width         = static_cast<int>(w);
        height        = static_cast<int>(h);
        bytesPerPixel = static_cast<int>(bpp);
        return true;
    }

    void VulkanRendererCore::setObjectInstanceId(const Object3D& obj, uint32_t instanceId) {
        core()->instanceIdOverride_[obj.id] = static_cast<uint16_t>(instanceId & 0xFFFFu);
    }

    void VulkanRendererCore::setObjectClassId(const Object3D& obj, uint32_t classId) {
        core()->classIds_[obj.id] = static_cast<uint16_t>(classId > 255u ? 255u : classId);
    }

    void VulkanRendererCore::setEventCameraEnabled(bool enabled) {
        auto& impl = *core();
        if (enabled == impl.eventCamEnabled_) return;
        impl.eventCamEnabled_ = enabled;
        if (enabled) {
            if (!impl.eventCam_) {
                impl.eventCam_ = std::make_unique<vulkan::EventCameraDetector>(*impl.ctx);
            }
            const VkExtent2D ext = impl.ctx->swapchainExtent();
            // Honour any user-pinned sensor resolution; 0 means "track
            // swapchain". Clamp to [16, swapchain] so we never dispatch
            // an empty grid or oversample the gbuf source.
            const uint32_t w = (impl.eventCamUserW_ == 0)
                    ? ext.width
                    : std::clamp(impl.eventCamUserW_, 16u, ext.width);
            const uint32_t h = (impl.eventCamUserH_ == 0)
                    ? ext.height
                    : std::clamp(impl.eventCamUserH_, 16u, ext.height);
            // Wait for any in-flight work that might reference the old
            // images before resize() destroys them.
            vkDeviceWaitIdle(impl.ctx->device());
            impl.eventCam_->resize(w, h);
            // Set up the deterministic shade pipeline — eliminates
            // stochastic shading noise as a source of false events. The
            // detector now reads the shade output (eventLumaBuf_) instead of
            // the scene capture buffer, so we don't need to enable
            // sceneCapture for the event camera to work.
            impl.createEventShadePipeline();
            impl.allocateEventLumaBuffer(w, h);
        }
    }

    bool VulkanRendererCore::eventCameraEnabled() const {
        return core()->eventCamEnabled_;
    }

    void VulkanRendererCore::setEventCameraParams(const EventCameraParams& p) {
        auto& impl = *core();
        impl.eventCamParams_.threshold         = p.threshold;
        impl.eventCamParams_.decay             = p.decay;
        impl.eventCamParams_.minLuma           = p.minLuma;
        impl.eventCamParams_.maxEventsPerPixel = p.maxEventsPerPixel;
        impl.eventCamParams_.frameTimeUs       = p.frameTimeUs;
    }

    VulkanRendererCore::EventCameraParams VulkanRendererCore::eventCameraParams() const {
        const auto& src = core()->eventCamParams_;
        EventCameraParams p{};
        p.threshold         = src.threshold;
        p.decay             = src.decay;
        p.minLuma           = src.minLuma;
        p.maxEventsPerPixel = src.maxEventsPerPixel;
        p.frameTimeUs       = src.frameTimeUs;
        return p;
    }

    std::vector<unsigned char> VulkanRendererCore::readEventCameraVisualisation() const {
        const auto& impl = *core();
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return {};
        return impl.eventCam_->readVisualisation();
    }

    size_t VulkanRendererCore::readEventCameraVisualisationInto(unsigned char* dst, size_t cap) const {
        const auto& impl = *core();
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return 0;
        return impl.eventCam_->readVisualisationInto(dst, cap);
    }

    size_t VulkanRendererCore::readEventStreamInto(Event* dst, size_t cap,
                                               bool* overflowed) const {
        const auto& impl = *core();
        if (!impl.eventCamEnabled_ || !impl.eventCam_) {
            if (overflowed) *overflowed = false;
            return 0;
        }
        // Public Event and detector Event are layout-identical (both 16B
        // {x, y, polarity, t_us}); reinterpret is safe and avoids any
        // per-event marshalling cost.
        static_assert(sizeof(Event) == sizeof(vulkan::EventCameraDetector::Event),
                      "Public Event must match detector Event byte-for-byte");
        return impl.eventCam_->readEventStreamInto(
                reinterpret_cast<vulkan::EventCameraDetector::Event*>(dst),
                cap, overflowed);
    }

    void VulkanRendererCore::setEventsOnlyMode(bool enabled) {
        core()->eventsOnlyMode_ = enabled;
    }

    bool VulkanRendererCore::eventsOnlyMode() const {
        return core()->eventsOnlyMode_;
    }

    void VulkanRendererCore::setEventCameraResolution(uint32_t width, uint32_t height) {
        auto& impl = *core();
        impl.eventCamUserW_ = width;
        impl.eventCamUserH_ = height;
        if (!impl.eventCamEnabled_ || !impl.eventCam_) return;

        // Effective sensor dims: zero means "track swapchain"; otherwise
        // clamp to [16, swapchain] so we never request 0-pixel dispatches
        // or values larger than the gbuf can possibly source.
        const VkExtent2D ext = impl.ctx->swapchainExtent();
        uint32_t w = (width  == 0) ? ext.width  : std::clamp(width,  16u, ext.width);
        uint32_t h = (height == 0) ? ext.height : std::clamp(height, 16u, ext.height);

        vkDeviceWaitIdle(impl.ctx->device());
        impl.eventCam_->resize(w, h);
        impl.allocateEventLumaBuffer(w, h);
    }

    std::pair<uint32_t, uint32_t> VulkanRendererCore::eventCameraResolution() const {
        const auto& impl = *core();
        if (!impl.eventCam_) return {impl.eventCamUserW_, impl.eventCamUserH_};
        return {impl.eventCam_->width(), impl.eventCam_->height()};
    }

    void VulkanRendererCore::dispose() { disposeImpl(); }

    void* VulkanRendererCore::nativeInstance() const {
        return static_cast<void*>(core()->ctx->instance());
    }
    void* VulkanRendererCore::nativePhysicalDevice() const {
        return static_cast<void*>(core()->ctx->physicalDevice());
    }
    void* VulkanRendererCore::nativeDevice() const {
        return static_cast<void*>(core()->ctx->device());
    }
    void* VulkanRendererCore::nativeGraphicsQueue() const {
        return static_cast<void*>(core()->ctx->graphicsQueue());
    }
    uint32_t VulkanRendererCore::graphicsQueueFamily() const {
        return core()->ctx->queueFamilies().graphics;
    }
    uint32_t VulkanRendererCore::nativeSwapchainFormat() const {
        return static_cast<uint32_t>(core()->ctx->swapchainFormat());
    }
    uint32_t VulkanRendererCore::imageCount() const {
        return static_cast<uint32_t>(core()->ctx->swapchainImages().size());
    }

    void VulkanRendererCore::setOverlayCallback(std::function<void(void*)> callback) {
        core()->overlayCallback = std::move(callback);
    }

    void VulkanRendererCore::setFogAnisotropy(float g) {
        g = std::max(-0.95f, std::min(g, 0.95f));
        if (g != core()->fogAnisotropy_) {
            core()->fogAnisotropy_ = g;
        }
    }

    float VulkanRendererCore::getFogAnisotropy() const {
        return core()->fogAnisotropy_;
    }

    void VulkanRendererCore::setFogWaterSurfaceY(float y) {
        core()->fogWaterSurfaceY_ = y;
    }

    void VulkanRendererCore::setRenderScale(float scale) {
        core()->setRenderScale(scale);
    }

    float VulkanRendererCore::renderScale() const {
        return core()->renderScale_;
    }

    void VulkanRendererCore::setDenoise(bool enabled) {
        core()->denoiseEnabled_ = enabled;
    }

    bool VulkanRendererCore::denoise() const {
        return core()->denoiseEnabled_;
    }

    void VulkanRendererCore::setBloomIntensity(float intensity) {
        core()->bloomIntensity_ = intensity < 0.f ? 0.f : intensity;
    }

    float VulkanRendererCore::bloomIntensity() const {
        return core()->bloomIntensity_;
    }

    void VulkanRendererCore::setDeferredDenoise(bool enabled) {
        core()->denoiseEnabled_ = enabled;
    }

    bool VulkanRendererCore::deferredDenoise() const {
        return core()->denoiseEnabled_;
    }

    void VulkanRenderer::setDeferredAO(bool enabled) {
        pimpl_->deferredAO_ = enabled;
    }

    bool VulkanRenderer::deferredAO() const {
        return pimpl_->deferredAO_;
    }

    void VulkanRenderer::setSsrReflections(bool enabled) {
        pimpl_->deferredSsr_ = enabled;
    }

    bool VulkanRenderer::ssrReflections() const {
        return pimpl_->deferredSsr_;
    }

    void VulkanRenderer::setProbeGI(bool enabled) {
        if (pimpl_->probeGIEnabled_ == enabled) return;
        pimpl_->probeGIEnabled_ = enabled;
        // Force a re-fit + SH clear on (re-)enable so a scene swap while the
        // feature was off can't leave the grid over stale bounds.
        if (enabled) pimpl_->probeGridDirty_ = true;
        // The probe term feeds the accumulated GI channel — toggling shifts
        // its converged mean, so reset accumulation to make the change land
        // immediately (same pattern as setRestirDIEnabled).
        if (pimpl_->sceneBuilt_) pimpl_->resetAccumulation();
    }

    bool VulkanRenderer::probeGI() const {
        return pimpl_->probeGIEnabled_;
    }

    void VulkanRenderer::setGbufferMsaa(uint32_t samples) {
        pimpl_->setGbufferMsaa(samples);
    }

    uint32_t VulkanRenderer::gbufferMsaa() const {
        return pimpl_->gbufferMsaa();
    }

    void VulkanRendererCore::setBloomThreshold(float threshold) {
        core()->bloomThreshold_ = threshold < 0.f ? 0.f : threshold;
    }

    float VulkanRendererCore::bloomThreshold() const {
        return core()->bloomThreshold_;
    }

    void VulkanRendererCore::setBloomClamp(float clampMax) {
        core()->bloomClamp_ = clampMax < 0.f ? 0.f : clampMax;
    }

    float VulkanRendererCore::bloomClamp() const {
        return core()->bloomClamp_;
    }

    void VulkanRendererCore::setSharpenStrength(float amount) {
        core()->sharpenStrength_ = amount < 0.f ? 0.f : amount;
    }

    float VulkanRendererCore::sharpenStrength() const {
        return core()->sharpenStrength_;
    }

    void VulkanRendererCore::setMotionBlur(float shutterFraction) {
        auto* c = core();
        const float clamped = std::clamp(shutterFraction, 0.f, 1.f);
        if (c->motionBlurAmount_ == clamped) return;
        const bool wasActive = c->motionBlurAmount_ > 0.f;
        c->motionBlurAmount_ = clamped;
        // In HDR-input mode, PostComposite's HDR-scene binding must switch
        // between the plain history slot and the motion-blur intermediate
        // (see rewriteBloomDescriptors' hdrSceneViews) depending on whether
        // blur is active. Don't touch GPU resources here (setter-side
        // allocation/descriptor writes are the pre-first-render crash class
        // — see feedback_vulkan_pre_first_render_setters.md); just flag it
        // dirty and let the next frame's ensureHybridResources (which also
        // owns the HDR-mode lazy allocation) pick it up and rewrite.
        if (c->taaHdrInput_ && wasActive != (clamped > 0.f)) {
            c->taaHdrPlumbingDirty_ = true;
        }
    }

    float VulkanRendererCore::motionBlur() const {
        return core()->motionBlurAmount_;
    }

    void VulkanRendererCore::setTaaHdrInput(bool enabled) {
        auto* c = core();
        if (c->taaHdrInput_ == enabled) return;
        c->taaHdrInput_ = enabled;
        // Switching domains mid-run would blend history written in one
        // colour space against neighbourhood stats computed in the other —
        // drop it (alpha=1 on the next resolve) exactly like a resize.
        // invalidateHistory is a cheap flag flip (no GPU work), safe here.
        if (c->taa_) c->taa_->invalidateHistory();
        // The HDR-mode-only scratch images (TaaResolve::mblurOutHdr_ /
        // PostComposite::hdrOut_) are allocated lazily and their descriptors
        // rewritten by ensureHybridResources on the NEXT frame — never
        // synchronously here. Enabling before the first render or mid-run
        // both land in the same deferred path, so the toggle engages one
        // frame later (mirrors setOcclusionCulling's farthest-HiZ pyramid).
        c->taaHdrPlumbingDirty_ = true;
    }

    bool VulkanRendererCore::taaHdrInput() const {
        return core()->taaHdrInput_;
    }

    void VulkanRendererCore::setPhysicalCamera(bool enabled) {
        core()->physicalCamera_ = enabled;
    }

    bool VulkanRendererCore::physicalCamera() const {
        return core()->physicalCamera_;
    }

    void VulkanRendererCore::setCameraExposure(float aperture, float shutterSeconds, float iso) {
        core()->camAperture_ = std::max(aperture, 0.1f);
        core()->camShutter_  = std::max(shutterSeconds, 1e-6f);
        core()->camIso_      = std::max(iso, 1.f);
    }

    void VulkanRendererCore::setExposureCompensation(float ev) {
        core()->camEvComp_ = std::clamp(ev, -20.f, 20.f);
    }

    float VulkanRendererCore::exposureCompensation() const {
        return core()->camEvComp_;
    }

    void VulkanRendererCore::setPhysicalLightUnits(bool enabled) {
        core()->physicalLightUnits_ = enabled;
    }

    bool VulkanRendererCore::physicalLightUnits() const {
        return core()->physicalLightUnits_;
    }

    void VulkanRendererCore::setDepthOfField(bool enabled) {
        core()->dofEnabled_ = enabled;
    }

    bool VulkanRendererCore::depthOfField() const {
        return core()->dofEnabled_;
    }

    void VulkanRendererCore::setFocusDistance(float meters) {
        core()->focusDistance_ = std::max(meters, 0.01f);
    }

    float VulkanRendererCore::focusDistance() const {
        return core()->focusDistance_;
    }

    void VulkanRendererCore::setOcclusionCulling(bool enabled) {
        auto* c = core();
        if (c->occlusionCullingEnabled_ == enabled) return;
        c->occlusionCullingEnabled_ = enabled;
        c->occlActiveThisFrame_ = false;// next buildIndirectDrawData re-evaluates
        // The farthest pyramid's image is allocated lazily by the next
        // frame's ensureHybridResources (which inherits the resize/MSAA
        // idle-wait guarantees), so enabling mid-run engages one frame later.
    }

    bool VulkanRendererCore::occlusionCulling() const {
        return core()->occlusionCullingEnabled_;
    }

    void VulkanRendererCore::setWhiteBalance(float temperatureK, float tint) {
        if (core()->post_) core()->post_->setWhiteBalance(temperatureK, tint);
    }

    void VulkanRendererCore::setColorGrade(const ColorGrade& grade) {
        if (!core()->post_) return;
        vulkan::PostComposite::ColorGrade g;
        g.lift[0]  = grade.lift.x;  g.lift[1]  = grade.lift.y;  g.lift[2]  = grade.lift.z;
        g.gamma[0] = grade.gamma.x; g.gamma[1] = grade.gamma.y; g.gamma[2] = grade.gamma.z;
        g.gain[0]  = grade.gain.x;  g.gain[1]  = grade.gain.y;  g.gain[2]  = grade.gain.z;
        g.saturation = grade.saturation;
        g.contrast   = grade.contrast;
        core()->post_->setColorGrade(g);
    }

    void VulkanRendererCore::setFireflyClamp(float cap) {
        core()->fireflyClamp_ = (cap <= 0.0f) ? 1e30f : cap;
    }

    float VulkanRendererCore::fireflyClamp() const {
        const float v = core()->fireflyClamp_;
        return (v > 1e20f) ? 0.0f : v;
    }

    void VulkanRendererCore::setSunAngularRadius(float degrees) {
        core()->sunAngularRadiusDeg_ = std::max(0.f, degrees);
    }

    float VulkanRendererCore::sunAngularRadius() const {
        return core()->sunAngularRadiusDeg_;
    }

    VulkanRendererCore::SoftBodyInteropHandle
    VulkanRendererCore::enableSoftBodyInterop(const Mesh& mesh, std::function<void()> deviceCopy) {
        return core()->enableSoftBodyInterop(mesh, std::move(deviceCopy));
    }

    void VulkanRenderer::setDeferredVolumetrics(float density, float anisotropy) {
        pimpl_->deferredVolDensity_ = std::max(density, 0.f);
        pimpl_->deferredVolAniso_   = std::clamp(anisotropy, -0.95f, 0.95f);
    }

    void VulkanRenderer::setVolumetricFog(bool enabled) {
        if (enabled != pimpl_->deferredVolFog_) {
            pimpl_->deferredVolFog_ = enabled;
        }
    }

    bool VulkanRenderer::volumetricFog() const {
        return pimpl_->deferredVolFog_;
    }

    void VulkanRenderer::setClouds(const std::optional<CloudSettings>& settings) {
        if (settings) {
            pimpl_->cloudsEnabled_    = true;
            pimpl_->cloudCoverage_    = std::clamp(settings->coverage, 0.f, 1.f);
            pimpl_->cloudDensity_     = std::max(settings->density, 0.f);
            pimpl_->cloudBottomY_     = settings->bottomY;
            pimpl_->cloudTopY_        = std::max(settings->topY, settings->bottomY + 1.f);
            pimpl_->cloudEvolveSpeed_ = std::max(settings->evolveSpeed, 0.f);
            pimpl_->cloudWind_[0]     = settings->wind.x;
            pimpl_->cloudWind_[1]     = settings->wind.y;
            pimpl_->cloudWind_[2]     = settings->wind.z;
        } else {
            pimpl_->cloudsEnabled_ = false;
        }
    }

    std::optional<VulkanRenderer::CloudSettings> VulkanRenderer::clouds() const {
        if (!pimpl_->cloudsEnabled_) return std::nullopt;
        CloudSettings s;
        s.coverage    = pimpl_->cloudCoverage_;
        s.density     = pimpl_->cloudDensity_;
        s.bottomY     = pimpl_->cloudBottomY_;
        s.topY        = pimpl_->cloudTopY_;
        s.evolveSpeed = pimpl_->cloudEvolveSpeed_;
        s.wind        = Vector3(pimpl_->cloudWind_[0], pimpl_->cloudWind_[1], pimpl_->cloudWind_[2]);
        return s;
    }

    void VulkanRenderer::setHeightFog(const std::optional<HeightFogSettings>& settings) {
        if (settings) {
            pimpl_->heightFogEnabled_     = true;
            pimpl_->heightFogDensity_     = std::max(settings->density, 0.f);
            pimpl_->heightFogBaseY_       = settings->baseY;
            pimpl_->heightFogFalloff_     = std::max(settings->falloff, 1.f);
            pimpl_->heightFogNoiseAmount_ = std::clamp(settings->noiseAmount, 0.f, 1.f);
        } else {
            pimpl_->heightFogEnabled_ = false;
        }
    }

    std::optional<VulkanRenderer::HeightFogSettings> VulkanRenderer::heightFog() const {
        if (!pimpl_->heightFogEnabled_) return std::nullopt;
        HeightFogSettings s;
        s.density     = pimpl_->heightFogDensity_;
        s.baseY       = pimpl_->heightFogBaseY_;
        s.falloff     = pimpl_->heightFogFalloff_;
        s.noiseAmount = pimpl_->heightFogNoiseAmount_;
        return s;
    }

    void VulkanRenderer::setDeferredStarfield(float intensity) {
        pimpl_->deferredStarIntensity_ = std::max(intensity, 0.f);
    }

    void VulkanRenderer::setEnvSunPolicy(EnvSunPolicy policy) {
        if (pimpl_->envSunPolicy_ == policy) return;
        const bool wasOff = pimpl_->envSunPolicy_ == EnvSunPolicy::Off;
        const bool isOff  = policy == EnvSunPolicy::Off;
        pimpl_->envSunPolicy_ = policy;
        // Auto↔Always only changes the INJECTION, which updateLightsUbo
        // re-evaluates every frame (its UBO hash change resets accumulation).
        // Off transitions change the PMREM content, so force the env re-upload
        // path (refreshEnvTextureFromScene early-outs on a matching texture id)
        // so the mips are rebuilt with/without the sun.
        if (wasOff != isOff) {
            pimpl_->envTextureIdUploaded = 0xFFFFFFFFu;
            pimpl_->envSun_ = {};
        }
    }

    VulkanRenderer::EnvSunPolicy VulkanRenderer::envSunPolicy() const {
        return pimpl_->envSunPolicy_;
    }

    void VulkanRenderer::setEnvSunExtraction(bool enabled) {
        setEnvSunPolicy(enabled ? EnvSunPolicy::Auto : EnvSunPolicy::Off);
    }

    bool VulkanRenderer::envSunExtraction() const {
        return pimpl_->envSunPolicy_ != EnvSunPolicy::Off;
    }

    bool VulkanRenderer::envSunFound() const {
        return pimpl_->envSun_.found;
    }

    Vector3 VulkanRenderer::envSunDirection() const {
        const auto& s = pimpl_->envSun_;
        return {s.dir[0], s.dir[1], s.dir[2]};
    }

    Vector3 VulkanRenderer::envSunColor() const {
        const auto& s = pimpl_->envSun_;
        return {s.colorE[0], s.colorE[1], s.colorE[2]};
    }

    void VulkanRenderer::setAutoExposure(bool enabled) {
        pimpl_->autoExposureEnabled_ = enabled;
    }

    bool VulkanRenderer::autoExposure() const {
        return pimpl_->autoExposureEnabled_;
    }

    void VulkanRenderer::setAutoExposureSpeed(float evPerSecond) {
        pimpl_->autoExpSpeed_ = std::max(evPerSecond, 0.01f);
        if (pimpl_->autoExposure_) pimpl_->autoExposure_->adaptSpeed = pimpl_->autoExpSpeed_;
    }

    void VulkanRenderer::setAutoExposureRange(float minEV, float maxEV) {
        pimpl_->autoExpMinEV_ = minEV;
        pimpl_->autoExpMaxEV_ = maxEV;
        if (pimpl_->autoExposure_) {
            pimpl_->autoExposure_->minEV = minEV;
            pimpl_->autoExposure_->maxEV = maxEV;
        }
    }

    void VulkanRendererCore::disableSoftBodyInterop(const Mesh& mesh) {
        core()->disableSoftBodyInterop(mesh);
    }

    void VulkanRendererCore::setRestirDIEnabled(bool enabled) {
        if (core()->restirDIEnabled_ == enabled) return;
        core()->restirDIEnabled_ = enabled;
        // ReSTIR is unbiased, so toggling it changes the convergence rate, not
        // the converged mean — on a settled frame the running-mean accumulator
        // would hide the switch entirely. Reset accumulation so the change is
        // actually visible. Gated on sceneBuilt_: before the first render there
        // is nothing accumulated and the gbuf/reservoir images aren't allocated
        // yet (clearGbufImages would touch null handles).
        if (core()->sceneBuilt_) core()->resetAccumulation();
    }

    bool VulkanRendererCore::restirDIEnabled() const {
        return core()->restirDIEnabled_;
    }

    void VulkanRendererCore::setNormalMapToksvig(bool enabled) {
        core()->normalMapToksvig_ = enabled;
    }

    bool VulkanRendererCore::normalMapToksvig() const {
        return core()->normalMapToksvig_;
    }

    void VulkanRendererCore::setAutoLod(bool enabled) {
        core()->autoLod_ = enabled;
    }

    bool VulkanRendererCore::autoLod() const {
        return core()->autoLod_;
    }

    VulkanRendererCore::AutoLodStats VulkanRendererCore::autoLodStats() const {
        return core()->autoLodStats_;
    }

    VulkanRendererCore::FrameTimings VulkanRendererCore::lastFrameTimings() const {
        return core()->gpuTimings_->timings();
    }

    void VulkanRendererCore::scanLidar(const std::vector<LidarBeam>& beams,
                                   std::vector<LidarReturn>& results,
                                   const LidarParams& params) {
        core()->scanLidar(beams, results, params);
    }

    void VulkanRendererCore::setOverlayLayer(int channel) {
        core()->overlayLayer_ = (channel < 0 || channel > 31) ? -1 : channel;
    }

    int VulkanRendererCore::overlayLayer() const {
        return core()->overlayLayer_;
    }

    void VulkanRendererCore::setHybridDebugView(int view) {
        using V = CoreImpl::HybridDebugView;
        switch (view) {
            case 1:  core()->hybridDebugView_ = V::Normal; break;
            case 2:  core()->hybridDebugView_ = V::Motion; break;
            case 3:  core()->hybridDebugView_ = V::Ids;    break;
            case 4:  core()->hybridDebugView_ = V::Albedo; break;
            case 5:  core()->hybridDebugView_ = V::Depth;  break;
            default: core()->hybridDebugView_ = V::Off;    break;
        }
    }

    int VulkanRendererCore::hybridDebugView() const {
        using V = CoreImpl::HybridDebugView;
        switch (core()->hybridDebugView_) {
            case V::Normal: return 1;
            case V::Motion: return 2;
            case V::Ids:    return 3;
            case V::Albedo: return 4;
            case V::Depth:  return 5;
            default:        return 0;
        }
    }

} // namespace threepp
