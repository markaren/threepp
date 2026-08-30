// VulkanCoreInit — VulkanRenderer::Impl's constructor and destructor.
//
// These two are the renderer's whole lifetime in one place: the ctor brings up
// the context, the per-frame rings, the passes and the primary view; the dtor
// tears the same list down in the reverse order. They lived inline in
// VulkanCoreImpl.hpp, where their 450 lines sat between the member
// declarations and made the header unreadable — and where a resource added to
// one half but not the other is easy to miss. Side by side in one TU, the two
// lists can be read against each other.

#include "VulkanCoreImpl.hpp"

namespace threepp {


VulkanRenderer::Impl::Impl(Canvas& c) : canvas(c), size(c.size()), lastCanvasSize(c.size()) {
            // The PRIMARY view. Created before anything else touches a
            // per-view resource, because view() dereferences curView_
            // unconditionally — every createXxx below allocates into it.
            views_.push_back(std::make_unique<ViewContext>());
            curView_ = views_[0].get();

            ctx = std::make_unique<VulkanContext>(
                    static_cast<GLFWwindow*>(canvas.windowPtr()),
                    /*enableRayTracing*/ true,
                    /*vsync*/ canvas.vsync(),
                    /*preferHeadlessSurface*/ canvas.headless());

            // Pin `size` to the extent the surface actually granted. The
            // canvas's size is what was asked for; the swapchain's is what
            // exists, and every pixel this renderer hands back (readRGBPixels,
            // writeFramebuffer) is swapchain-shaped. recreateSwapchainAndDescriptors
            // already reconciles the two after a resize — do it at birth too,
            // so a window the platform clamped on creation never gets a frame
            // read back against the size it failed to get.
            if (const VkExtent2D ext = ctx->swapchainExtent(); ext.width > 0 && ext.height > 0) {
                size = WindowSize{static_cast<int>(ext.width), static_cast<int>(ext.height)};
            }

            // The scene-dependent AS build runs lazily on the first render()
            // call. Everything below is scene-independent and safe at ctor time.
            createCommandResources();
            createCameraUbos();
            createLightsUbos();
            createFogUbos();
            createCloudUbos();
            // EnvPrefilter owns the PMREM compute pipeline + descriptor pool.
            // Construct before createDefaultEnvImage so the env upload path is
            // ready if scene.environment is set before the first render().
            envPrefilter_ = std::make_unique<vulkan::EnvPrefilter>(*ctx, cmdPool);
            createDefaultEnvImage();
            createTextureSampler();
            createDefaultMaterialTexture();
            // Allocates the ReSTIR DI reservoir ping-pong images the deferred
            // shade's NEE consumes.
            createReservoirImages();
            skinning_ = std::make_unique<vulkan::SkinningPipeline>(*ctx);
            tetSkinning_ = std::make_unique<vulkan::TetSkinningPipeline>(*ctx);
            waterDisplace_ = std::make_unique<vulkan::WaterDisplacePipeline>(*ctx);
            foamWorld_     = std::make_unique<vulkan::FoamWorldPipeline>(*ctx);
            grassWind_     = std::make_unique<vulkan::GrassWindPipeline>(*ctx);
            // Hybrid raster G-buffer infrastructure. Costs a few hundred MB
            // at 1080p for six attachments × kFramesInFlight.
            ensureHybridResources();
            // TAA pipeline + images. The deferred shade's descriptor binding 1
            // (shade output target) always points at the TAA input view.
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            view().taa_ = std::make_unique<vulkan::TaaResolve>(
                    *ctx, cmdPool, imageCount_, kFramesInFlight);
            {
                // TAA input is the deferred render extent; history +
                // output are the swapchain extent. When they differ the
                // resolve pass runs as a temporal upsampler.
                const VkExtent2D inExt  = renderExtent();
                const VkExtent2D outExt = viewOutExtent();
                view().taa_->createImages(inExt.width, inExt.height,
                                   outExt.width, outExt.height);
            }
#if defined(THREEPP_WITH_DLSS)
            // DLSS feature at the display extent. Created BEFORE FSR because DLSS
            // OUTRANKS it (useFsr() requires !useDlss()): an FSR context built
            // alongside an active DLSS feature could never dispatch, so we skip it
            // below and save its context (~60 MB committed here). NGX feature
            // creation records GPU init work — one-shot submit+wait. On NGX
            // failure (non-RTX GPU / old driver) dlssActive_ stays false and the
            // FSR/TAA fallback runs.
            dlss_ = std::make_unique<vulkan::DlssUpscaler>(*ctx, kFramesInFlight);
            {
                VkCommandBuffer initCb = beginOneShot();
                dlssActive_ = dlss_->create(initCb,
                                            ctx->swapchainExtent().width,
                                            ctx->swapchainExtent().height,
                                            renderExtent().width,
                                            renderExtent().height);
                endAndSubmitOneShot(initCb, "DLSS feature create");
            }
            dlssResetNext_ = true;
#endif
#if defined(THREEPP_WITH_FSR)
            // FSR upscaler context at the display (swapchain) extent. FSR stores
            // the display extent at create time but takes the render extent per
            // dispatch, so a renderScale change needs no recreation — only a
            // swapchain/display resize does (handled in reallocateRenderExtentResources).
            // On success FSR replaces the TAA resolve; on failure the TAA path
            // runs unchanged. Skipped entirely when DLSS already claimed the
            // upscaler slot (it could never dispatch); if DLSS is disabled at
            // runtime, setDlss(false) creates the FSR context on demand.
            bool dlssClaimedUpscaler = false;
#if defined(THREEPP_WITH_DLSS)
            dlssClaimedUpscaler = dlssActive_;
#endif
            if (!dlssClaimedUpscaler) {
                fsr_ = std::make_unique<vulkan::FsrUpscaler>(*ctx, kFramesInFlight);
                fsrActive_ = fsr_->create(ctx->swapchainExtent().width,
                                          ctx->swapchainExtent().height);
                fsrResetNext_ = true;
            }
#endif
            // HDR bloom pyramid. sceneHdr lives at the deferred render
            // extent (it is the shared set's binding 1 target); the bloom
            // pyramid levels are half that and below.
            view().bloom_ = std::make_unique<vulkan::BloomPass>(*ctx, cmdPool, kFramesInFlight);
            view().bloom_->createImages(renderExtent().width, renderExtent().height);
            onAfterBloomCreateImages();
            // Exposure/WB/tone-map/grade/sRGB composite → TAA input.
            view().post_ = std::make_unique<vulkan::PostComposite>(*ctx, cmdPool, kFramesInFlight);
            // Thin-lens DoF (images/descriptors fitted in rewriteBloomDescriptors).
            dof_ = std::make_unique<vulkan::DofPass>(*ctx, cmdPool, kFramesInFlight);
            // Gaussian splats (images/descriptors fitted in rewriteBloomDescriptors,
            // like DoF; buffers only appear once a scene actually has a cloud).
            splat_ = std::make_unique<vulkan::SplatPass>(*ctx, cmdPool, kFramesInFlight);
            // Lens distortion + sensor noise, applied to the FINISHED frame at
            // the very end of recording (after the overlay pass — see
            // SensorPass.hpp). Allocates nothing until a lens or noise is set.
            sensorPass_ = std::make_unique<vulkan::SensorPass>(*ctx, cmdPool, kFramesInFlight);
            // Raster-first deferred lighting pass. Writes bloom_->sceneHdr, so
            // it must exist after bloom_; its descriptors reference the camera /
            // lights UBOs, the env image, the raster gbuffer and sceneHdr — all
            // created above by this point.
            // The deferred base traces ray-query shadow rays, so only stand it
            // up when the device supports VK_KHR_ray_query. Without it,
            // deferredShade_ stays null if ray query is unavailable.
            if (ctx->rayQuerySupported()) {
                view().deferredShade_ = std::make_unique<vulkan::DeferredShade>(*ctx, kFramesInFlight);
                // Probe grid backs the deferred set's bindings 36/37 (dummy-
                // free: real buffers from construction, sampling gated by the
                // grid UBO's enable flag until setProbeGI(true)).
                probeGI_ = std::make_unique<vulkan::ProbeGI>(*ctx, kFramesInFlight);
            }
            createBlueNoiseImage_();// must run before descriptor writes (binding 27)
            createOceanFineDummy_();// must run before descriptor writes (binding 32)
            createOceanFoamDummy_();// must run before descriptor writes (binding 33)
            createFoamDetailImage_();// must run before descriptor writes (binding 45 + deferred 34)
            rewriteTaaDescriptors();// after ensureHybridResources gave us raster gbuf views
            rewriteBloomDescriptors();// bloom composite reads gbuf + writes the TAA input
            rewriteDeferredDescriptors();// raster-first deferred shade inputs
            gpuTimings_ = std::make_unique<vulkan::GpuTimings>(*ctx, kFramesInFlight);
            overlayPass_ = std::make_unique<vulkan::OverlayPass>(
                    *ctx, kFramesInFlight,
                    // Atlas uploads are recorded into the frame's own cb (no
                    // one-shot submit + queue drain — that stalled on every
                    // in-flight frame each time a HUD TextSprite re-rasterized).
                    [this](VkCommandBuffer cb, uint32_t w, uint32_t h, VkFormat fmt,
                           const void* pix, VkDeviceSize sz,
                           VkFilter filter,
                           VkSamplerAddressMode addrU,
                           VkSamplerAddressMode addrV,
                           const char* name) {
                        return createSampledImage2DInFrame(cb, w, h, fmt, pix, sz,
                                                           filter, addrU, addrV, name);
                    },
                    // Retire stale sprite atlases through the frame-serial queue
                    // (no per-swap vkDeviceWaitIdle). Stamped with the frame
                    // being recorded when OverlayPass::record runs.
                    [this](Image2D&& img) { retire(std::move(img)); },
                    // Same for the overlay's line/sprite geometry buffers, which
                    // are re-uploaded and evicted mid-record while the previous
                    // frames-in-flight may still be drawing from them.
                    [this](Buffer&& b) { retire(std::move(b)); });

            // Optional one-shot fixed-footprint dump. Everything constructed above
            // is scene-independent, so this is the renderer's baseline cost — the
            // number to watch as new features add persistent targets. Enable with
            // THREEPP_VK_MEMDUMP=1 (or call dumpMemoryStats() from app code).
            if (const char* e = std::getenv("THREEPP_VK_MEMDUMP"); e && *e && *e != '0') {
                dumpMemoryStats("post-init");
            }
        }

VulkanRenderer::Impl::~Impl() {
            // Stop the auto-LOD background worker first — it's pure CPU (no
            // Vulkan handles touched), so this is safe even when ctx is
            // null. Must happen before the blasCache teardown below, which
            // destroys the same records' lodLevels AS/buffers the worker
            // could otherwise still be racing to enqueue results against
            // (drainLodResults only ever runs from ensureSceneBuilt, which
            // can't run concurrently with this destructor, but the queue
            // itself is shared state the worker thread still owns until
            // joined).
            {
                std::lock_guard<std::mutex> lk(lodJobMutex_);
                lodWorkerStop_ = true;
            }
            lodJobCv_.notify_all();
            if (lodWorker_.joinable()) lodWorker_.join();

            if (!ctx) return;
            VkDevice d = ctx->device();
            // If a frame was mid-record when the renderer was destroyed (the
            // canvas frame-end callback never got to fire, or the user tore
            // down the renderer from inside the animate lambda), close the
            // open cmd buffer without submitting. The reset-but-unsignaled
            // fence is fine — no queued work depends on it — and the cmd
            // pool destroy below releases the buffer regardless.
            if (frameState_ != FrameState::Idle) {
                vkEndCommandBuffer(cmdBuffers[currentFrame]);
                frameState_ = FrameState::Idle;
            }
            vkDeviceWaitIdle(d);
            // Device idle ⇒ nothing references retired resources. Destroy them
            // now or they leak at device destroy (VUID-vkDestroyDevice-device-
            // 05137 — the class of bug that bit lineGeomCache_ below).
            flushRetireQueue();

            // Secondary views that were never removeView()'d. Their resources
            // (rasterGbuf images, colorTarget, readbackBuf, post chain) are
            // otherwise only freed by applyPendingViewChanges after a
            // removeView, so they leak at device destroy (VUID-vkDestroyDevice-
            // device-05137). destroyBuffer/destroyImage2D null their handles,
            // so the per-view loops further down become no-ops for these.
            // Same guard as applyPendingViewChanges: a view still marked
            // pendingCreate never got resources.
            curView_ = views_[0].get();
            for (size_t i = 1; i < views_.size(); ++i) {
                if (!views_[i]->pendingCreate)
                    destroySecondaryViewResources(*views_[i]);
            }

            for (auto s : imageAvailable) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto s : renderFinished) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto f : inFlight) if (f) vkDestroyFence(d, f, nullptr);
            if (cmdPool) vkDestroyCommandPool(d, cmdPool, nullptr);
            gpuTimings_.reset();// query pool destruction while device is still valid
            // Explicit, next to gpuTimings_ and for the same reason: the pass
            // frees its per-field position rings through ctx's allocator, and
            // the device is provably idle here (flushRetireQueue above).
            particleFieldPass_.reset();
            particleFields_.clear();
            // The two density handles the renderer (not the pass) owns.
            for (auto& b : particleDensityUbos_) destroyBuffer(ctx->allocator(), b);
            destroyImage2D(ctx->allocator(), d, particleDensityDummy_);
            destroyImage2D(ctx->allocator(), d, particleDensityLinDummy_);
            // The splat reflection table's two renderer-owned handles, beside
            // their density twins and for the same reason (the baked volumes
            // themselves belong to SplatPass and die with their clouds).
            for (auto& b : splatVolumeUbos_) destroyBuffer(ctx->allocator(), b);
            destroyImage2D(ctx->allocator(), d, splatVolumeDummy_);

            if (tlas) ctx->rt().destroyAccelerationStructure(d, tlas, nullptr);
            destroyBuffer(ctx->allocator(), tlasBuffer);
            for (auto& b : tlasInstancesBuffers) destroyBuffer(ctx->allocator(), b);
            destroyBuffer(ctx->allocator(), tlasRefitScratch_);
            for (auto& b : geometryDescsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : materialDescsBuffers) destroyBuffer(ctx->allocator(), b);
            destroyBuffer(ctx->allocator(), sceneCaptureBuf_);
            // Frame-interop exports (enableFrameInterop). Closing the Win32 NT
            // handles here is why teardown must come after the consumer has
            // released its CUDA imports — the documented order.
            destroyFrameInterops();
            destroyBuffer(ctx->allocator(), eventLumaBuf_);
            if (eventShadePipeline_)       vkDestroyPipeline(d, eventShadePipeline_, nullptr);
            if (eventShadePipelineLayout_) vkDestroyPipelineLayout(d, eventShadePipelineLayout_, nullptr);
            if (eventShadeDescPool_)       vkDestroyDescriptorPool(d, eventShadeDescPool_, nullptr);
            if (eventShadeDsLayout_)       vkDestroyDescriptorSetLayout(d, eventShadeDsLayout_, nullptr);
            if (debugResolvePipeline_)       vkDestroyPipeline(d, debugResolvePipeline_, nullptr);
            if (debugResolvePipelineLayout_) vkDestroyPipelineLayout(d, debugResolvePipelineLayout_, nullptr);
            if (debugResolveDescPool_)       vkDestroyDescriptorPool(d, debugResolveDescPool_, nullptr);
            if (debugResolveDsLayout_)       vkDestroyDescriptorSetLayout(d, debugResolveDsLayout_, nullptr);

            for (auto& [_, rec] : blasCache) {
                destroyBlasLodLevels(*rec);
                destroyBlasRecord(*rec);
            }
            blasCache.clear();

            for (auto& [_, st] : skinnedMeshStates) {
                // Destroy the GPU-skinning input buffers + scratch first;
                // BLAS buffers below.
                destroyBuffer(ctx->allocator(), st->baseVertex);
                destroyBuffer(ctx->allocator(), st->baseNormal);
                destroyBuffer(ctx->allocator(), st->skinIndex);
                destroyBuffer(ctx->allocator(), st->skinWeight);
                for (auto& slot : st->boneMatrices) destroyBuffer(ctx->allocator(), slot);
                destroyBuffer(ctx->allocator(), st->blasScratch);
                if (st->blas) destroyBlasRecord(*st->blas);
            }
            skinnedMeshStates.clear();

            for (auto& [_, st] : tetMeshStates) {
                destroyBuffer(ctx->allocator(), st->tetIndex);
                destroyBuffer(ctx->allocator(), st->tetWeight);
                destroyBuffer(ctx->allocator(), st->baseNormal);
                destroyBuffer(ctx->allocator(), st->restInv0);
                destroyBuffer(ctx->allocator(), st->restInv1);
                destroyBuffer(ctx->allocator(), st->restInv2);
                for (auto& slot : st->tetPos) destroyBuffer(ctx->allocator(), slot);
                vulkan::destroyExternalBuffer(d, st->tetPosExt);
                destroyBuffer(ctx->allocator(), st->blasScratch);
                if (st->blas) destroyBlasRecord(*st->blas);
            }
            tetMeshStates.clear();

            for (auto& [_, st] : displacedStates) {
                if (st->blas) destroyBlasRecord(*st->blas);
                if (st->scratchA.view  != VK_NULL_HANDLE) vkDestroyImageView(d, st->scratchA.view, nullptr);
                if (st->scratchA.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->scratchA.image, st->scratchA.alloc);
                if (st->foamImage.view  != VK_NULL_HANDLE) vkDestroyImageView(d, st->foamImage.view, nullptr);
                if (st->foamImage.image != VK_NULL_HANDLE) vmaDestroyImage(ctx->allocator(), st->foamImage.image, st->foamImage.alloc);
                for (auto& ring : st->heightReadback)
                    for (auto& b : ring) destroyBuffer(ctx->allocator(), b);
                destroyBuffer(ctx->allocator(), st->foamDisturbBuffer);
                destroyBuffer(ctx->allocator(), st->wakeTrailBuffer);
                // Per-cascade Phillips / DynamicSpectrum / IFFT are RAII; their
                // destructors handle their own VkImage / VkPipeline / DSet cleanup.
            }
            displacedStates.clear();

            for (auto& [_, st] : grassStates) {
                if (st->blas) destroyBlasRecord(*st->blas);
                destroyBuffer(ctx->allocator(), st->restPos);
                destroyBuffer(ctx->allocator(), st->heightFrac);
            }
            grassStates.clear();

            for (auto& [_, st] : morphedMeshStates) {
                if (st->blas) destroyBlasRecord(*st->blas);
            }
            morphedMeshStates.clear();

            // Per-view buffers/images: walk EVERY view, not just the current
            // one. Loops here (rather than view().x) so adding a secondary
            // view can never leak by forgetting to extend the destructor.
            for (auto& vp : views_) {
                for (auto& b : vp->cameraUbos)         destroyBuffer(ctx->allocator(), b);
                for (auto& b : vp->rasterCameraUbos)   destroyBuffer(ctx->allocator(), b);
                for (auto& b : vp->drawInfoBuffers)    destroyBuffer(ctx->allocator(), b);
                for (auto& b : vp->indirectCmdBuffers) destroyBuffer(ctx->allocator(), b);
                for (auto& img : vp->reservoirPosImagesPP) destroyImage2D(ctx->allocator(), d, img);
                for (auto& img : vp->reservoirWImagesPP)   destroyImage2D(ctx->allocator(), d, img);
            }
            for (auto& b : lightsUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : clusterLightsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : clusterGridBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : fogUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : cloudUbos) destroyBuffer(ctx->allocator(), b);
            for (auto& b : motionMatBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : meshMovedBitsBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& b : emissiveTriBuffers) destroyBuffer(ctx->allocator(), b);
            destroyImage2D(ctx->allocator(), d, envImage);
            destroyImage2D(ctx->allocator(), d, blueNoiseImage);
            // 1x1 stand-ins bound to the MS gbuffer descriptor slots whenever
            // MSAA is off. Created once by ensureGbufDummyMS() (guarded by
            // gbufDummyMSCreated_) and, until now, destroyed nowhere at all —
            // 5 VkImage + 5 VkImageView leaked per device. They are not
            // recreated on resize, so this destructor is their only owner.
            for (auto& img : gbufDummyMS_) destroyImage2D(ctx->allocator(), d, img);
            destroyImage2D(ctx->allocator(), d, oceanFineHeightDummy);
            destroyImage2D(ctx->allocator(), d, oceanFoamDummy);
            destroyImage2D(ctx->allocator(), d, foamDetailImage);
            for (auto& img : materialTextures) destroyImage2D(ctx->allocator(), d, img);
            materialTextures.clear();
            if (textureSampler_) vkDestroySampler(d, textureSampler_, nullptr);
            if (textureSamplerIso_) vkDestroySampler(d, textureSamplerIso_, nullptr);
            if (textureSamplerClamp_) vkDestroySampler(d, textureSamplerClamp_, nullptr);
            if (textureSamplerIsoClamp_) vkDestroySampler(d, textureSamplerIsoClamp_, nullptr);
            if (textureSamplerCustom_) vkDestroySampler(d, textureSamplerCustom_, nullptr);
            if (textureSamplerCustomClamp_) vkDestroySampler(d, textureSamplerCustomClamp_, nullptr);
            for (VkSampler s : parkedSamplers_) vkDestroySampler(d, s, nullptr);
            parkedSamplers_.clear();

            // EnvPrefilter owns its pipeline / layout / pool / sampler.
            envPrefilter_.reset();

            // GPU skinning teardown. Per-SkinnedMeshState buffers were already
            // destroyed alongside the BLAS in the skinnedMeshStates clear
            // (above); the shared pipeline + pool live in skinning_.
            skinning_.reset();
            tetSkinning_.reset();

            // Water displace pipeline owns its handles + sampler.
            waterDisplace_.reset();
            // World-space foam pipeline; foam ping-pong images are owned by
            // each DisplacedMeshState and destroyed there.
            foamWorld_.reset();
            // Grass-wind pipeline; per-mesh buffers freed in the grassStates
            // loop above.
            grassWind_.reset();
            // Vertex-interop sanitize pipeline (lazy — null unless a mesh armed
            // interop). Its descriptor sets live in BlasRecords, all of which the
            // blasCache/state teardown above has already destroyed, so the pool
            // goes down with no live set left in it.
            vertexSanitize_.reset();

            // Hybrid raster G-buffer cleanup. Resources are lazy-created on
            // first render(); if render() was never called, all handles stay
            // VK_NULL_HANDLE and these calls become no-ops.
            destroyRasterGbufImages();
            // (rasterCameraUbos / drawInfoBuffers / indirectCmdBuffers are
            //  freed in the per-view loop above, alongside the camera UBOs.)
            if (rasterGbufPipeline)         vkDestroyPipeline(d, rasterGbufPipeline, nullptr);
            if (rasterGbufIndirectPipeline) vkDestroyPipeline(d, rasterGbufIndirectPipeline, nullptr);
            if (rasterGbufParticlePipeline) vkDestroyPipeline(d, rasterGbufParticlePipeline, nullptr);
            if (rasterGbufDecalPipeline)    vkDestroyPipeline(d, rasterGbufDecalPipeline, nullptr);
            if (rasterPipelineLayout)   vkDestroyPipelineLayout(d, rasterPipelineLayout, nullptr);
            if (rasterDsLayout)         vkDestroyDescriptorSetLayout(d, rasterDsLayout, nullptr);
            for (auto& vp : views_)
                if (vp->rasterDescPool) vkDestroyDescriptorPool(d, vp->rasterDescPool, nullptr);
            if (rasterGbufRenderPass)   vkDestroyRenderPass(d, rasterGbufRenderPass, nullptr);
            if (occlRenderPassA_)       vkDestroyRenderPass(d, occlRenderPassA_, nullptr);
            if (occlRenderPassB_)       vkDestroyRenderPass(d, occlRenderPassB_, nullptr);
            // MSAA render passes + pipelines: the four rasterGbuf*MS handles used
            // to be leaked here (only the occl MS passes were destroyed). Same
            // owner as the runtime MSAA-off path now. Device is idle at this point.
            destroyRasterGbufMsObjects();
            if (overlayWireframePipeline)         vkDestroyPipeline(d, overlayWireframePipeline, nullptr);
            if (overlayBasicPipeline)             vkDestroyPipeline(d, overlayBasicPipeline, nullptr);
            if (overlayBasicTransparentPipeline)  vkDestroyPipeline(d, overlayBasicTransparentPipeline, nullptr);
            if (overlayLineListPipeline)          vkDestroyPipeline(d, overlayLineListPipeline, nullptr);
            if (overlayLineStripPipeline)         vkDestroyPipeline(d, overlayLineStripPipeline, nullptr);
            if (overlayLineListColoredPipeline)   vkDestroyPipeline(d, overlayLineListColoredPipeline, nullptr);
            if (overlayLineStripColoredPipeline)  vkDestroyPipeline(d, overlayLineStripColoredPipeline, nullptr);
            for (auto& byStrip : overlayLineBlendPipelines)
                for (auto& byMode : byStrip)
                    for (VkPipeline p : byMode)
                        if (p) vkDestroyPipeline(d, p, nullptr);
            for (VkPipeline p : overlayMeshColoredPipelines)
                if (p) vkDestroyPipeline(d, p, nullptr);
            if (overlayBasicAdditivePipeline)     vkDestroyPipeline(d, overlayBasicAdditivePipeline, nullptr);
            if (overlayPointListPipeline)         vkDestroyPipeline(d, overlayPointListPipeline, nullptr);
            if (overlayDepthPrepassPipeline)      vkDestroyPipeline(d, overlayDepthPrepassPipeline, nullptr);
            if (overlayPipelineLayout)      vkDestroyPipelineLayout(d, overlayPipelineLayout, nullptr);
            // Hardware-MSAA overlay resources.
            if (overlayInjectPipeline_)       vkDestroyPipeline(d, overlayInjectPipeline_, nullptr);
            if (overlayInjectPipelineLayout_) vkDestroyPipelineLayout(d, overlayInjectPipelineLayout_, nullptr);
            if (overlayInjectSetLayout_)      vkDestroyDescriptorSetLayout(d, overlayInjectSetLayout_, nullptr);
            if (overlayInjectPool_)           vkDestroyDescriptorPool(d, overlayInjectPool_, nullptr);
            // Splat depth stamp (created alongside the overlay pipelines).
            if (splatStampPipeline_)          vkDestroyPipeline(d, splatStampPipeline_, nullptr);
            if (splatStampPipelineLayout_)    vkDestroyPipelineLayout(d, splatStampPipelineLayout_, nullptr);
            if (splatStampSetLayout_)         vkDestroyDescriptorSetLayout(d, splatStampSetLayout_, nullptr);
            if (splatStampPool_)              vkDestroyDescriptorPool(d, splatStampPool_, nullptr);
            if (overlayMsColor_.image != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, overlayMsColor_);
            if (overlayMsDepth_.image != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, overlayMsDepth_);
            if (overlayAaScratch_.image != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, overlayAaScratch_);
            // Particle billboard pass resources.
            if (particlePipelineNormal_)    vkDestroyPipeline(d, particlePipelineNormal_, nullptr);
            if (particlePipelineAdditive_)  vkDestroyPipeline(d, particlePipelineAdditive_, nullptr);
            if (particlePipelineLayout_)    vkDestroyPipelineLayout(d, particlePipelineLayout_, nullptr);
            // ParticleField billboards (F-D). The 1x sibling ALIASES the main
            // pipeline unless overlay MSAA forced a second object — destroy it
            // only when it is genuinely a second object.
            if (fieldBillboardPipeline1xOwned_ && fieldBillboardPipeline1x_)
                vkDestroyPipeline(d, fieldBillboardPipeline1x_, nullptr);
            if (fieldBillboardPipeline_)    vkDestroyPipeline(d, fieldBillboardPipeline_, nullptr);
            // 4c: the alpha-over pair, same aliasing rule.
            if (fieldBillboardAlphaPipeline1xOwned_ && fieldBillboardAlphaPipeline1x_)
                vkDestroyPipeline(d, fieldBillboardAlphaPipeline1x_, nullptr);
            if (fieldBillboardAlphaPipeline_)
                vkDestroyPipeline(d, fieldBillboardAlphaPipeline_, nullptr);
            if (fieldBillboardGlowPipeline_) vkDestroyPipeline(d, fieldBillboardGlowPipeline_, nullptr);
            if (fieldBillboardPipelineLayout_)
                vkDestroyPipelineLayout(d, fieldBillboardPipelineLayout_, nullptr);
            // F4 billboard glow. The pass object owns its own images, pools and
            // compute pipelines; only the composite graphics pipeline lives out
            // here, because it needs the swapchain format the pass knows nothing
            // about.
            if (fieldGlowCompositePipeline_) vkDestroyPipeline(d, fieldGlowCompositePipeline_, nullptr);
            if (fieldGlowCompositeLayout_)
                vkDestroyPipelineLayout(d, fieldGlowCompositeLayout_, nullptr);
            billboardGlow_.reset();
            if (particleDescSetLayout_)     vkDestroyDescriptorSetLayout(d, particleDescSetLayout_, nullptr);
            for (auto& pool : particleDescPools_) {
                if (pool) vkDestroyDescriptorPool(d, pool, nullptr);
            }
            // Overlay-fog UBO resources (Phase 2b).
            if (overlayFogDescPool_)        vkDestroyDescriptorPool(d, overlayFogDescPool_, nullptr);
            if (overlayFogDescSetLayout_)   vkDestroyDescriptorSetLayout(d, overlayFogDescSetLayout_, nullptr);
            for (auto& b : overlayFogUbos_) destroyBuffer(ctx->allocator(), b);
            // Lit-particle buffers + IO sets (particle_light.comp).
            if (particleIoDescPool_)        vkDestroyDescriptorPool(d, particleIoDescPool_, nullptr);
            for (auto& b : particleCenterBufs_) destroyBuffer(ctx->allocator(), b);
            for (auto& b : particleLightBufs_)  destroyBuffer(ctx->allocator(), b);
            if (spriteWorldPipeline_)       vkDestroyPipeline(d, spriteWorldPipeline_, nullptr);
            destroyBuffer(ctx->allocator(), spriteQuadVtx_);
            destroyBuffer(ctx->allocator(), spriteQuadIdx_);
            if (particleWhiteTex_.view != VK_NULL_HANDLE)
                destroyImage2D(ctx->allocator(), d, particleWhiteTex_);
            for (auto& [t, rec] : particleTexCache_) {
                destroyImage2D(ctx->allocator(), d, rec.image);
            }
            particleTexCache_.clear();
            for (auto& [g, rec] : particleGeomCache_) {
                destroyParticleGeomRec(rec);
            }
            particleGeomCache_.clear();
            // 3D hybrid-overlay line/point geometry cache (GridHelper, AxesHelper,
            // live point clouds). Pre-existing: this Impl-level cache had no
            // teardown, so its vertex/index/color buffers leaked at device
            // destroy (VUID-vkDestroyDevice-device-05137) for any scene with a
            // Line/Points overlay. Mirror OverlayPass's own lineGeomCache_ cleanup.
            for (auto& [g, rec] : lineGeomCache_) {
                destroyBuffer(ctx->allocator(), rec.vertex);
                if (rec.index.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.index);
                if (rec.color.handle != VK_NULL_HANDLE) destroyBuffer(ctx->allocator(), rec.color);
            }
            lineGeomCache_.clear();
            overlayPass_.reset();// destroy sprite/line pipelines + caches while device is alive
            if (gbufSampler_)           vkDestroySampler(d, gbufSampler_, nullptr);
            destroyBuffer(ctx->allocator(), dummyUvBuffer_);

#if defined(THREEPP_WITH_FSR)
            // Destroy the ffx-api context while the Vulkan device is still alive
            // (before taa_/ctx unwind). No-op if FSR never created.
            fsr_.reset();
#endif
#if defined(THREEPP_WITH_DLSS)
            // Release the NGX feature + shut NGX down while the device is alive.
            dlss_.reset();
#endif
            // Per-view passes own pipelines, layouts, samplers, descriptor
            // pools and images, all of which need the device alive — so they
            // are released HERE rather than left to ~ViewContext, which runs
            // when views_ unwinds. Every view, not just the current one.
            for (auto& vp : views_) {
                vp->taa_.reset();
                vp->post_.reset();
                vp->bloom_.reset();
                vp->deferredShade_.reset();
            }
        }
}// namespace threepp
