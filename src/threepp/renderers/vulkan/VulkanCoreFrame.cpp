#include "VulkanCoreImpl.hpp"

#include "VulkanCpuPhaseProf.hpp"

namespace threepp {

void VulkanRenderer::Impl::createCommandResources() {
            VkCommandPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pci.queueFamilyIndex = ctx->queueFamilies().graphics;
            check(vkCreateCommandPool(ctx->device(), &pci, nullptr, &cmdPool),
                  "vkCreateCommandPool");

            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = kFramesInFlight;
            check(vkAllocateCommandBuffers(ctx->device(), &ai, cmdBuffers.data()),
                  "vkAllocateCommandBuffers");

            VkSemaphoreCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            for (uint32_t i = 0; i < kFramesInFlight; ++i) {
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &imageAvailable[i]), "vkCreateSemaphore A");
                check(vkCreateFence(ctx->device(), &fci, nullptr, &inFlight[i]), "vkCreateFence");
            }
            createRenderFinishedSemaphores();
        }

void VulkanRenderer::Impl::createRenderFinishedSemaphores() {
            for (auto s : renderFinished)
                if (s) vkDestroySemaphore(ctx->device(), s, nullptr);
            renderFinished.assign(ctx->swapchainImages().size(), VK_NULL_HANDLE);
            VkSemaphoreCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (auto& s : renderFinished)
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &s), "vkCreateSemaphore B");
            // New images ⇒ every pinned index is stale. Called from
            // createCommandResources and from every swapchain recreation, which
            // is exactly the set of moments that invalidate them.
            pinnedSwapImage_.fill(UINT32_MAX);
            acquireSemPending_.fill(false);
        }

VkResult VulkanRenderer::Impl::acquireOrReuseSwapchainImage(uint32_t& imageIndex) {
            if (ctx->presentSuppressed() && pinnedSwapImage_[currentFrame] != UINT32_MAX) {
                // This slot already owns an image and nothing ever took it back,
                // so there is nothing to acquire and nothing to wait for: the
                // inFlight fence the caller just waited on is what orders this
                // frame's writes after the previous frame that used this slot —
                // and therefore this same image.
                imageIndex = pinnedSwapImage_[currentFrame];
                return VK_SUCCESS;
            }
            const VkResult r = vkAcquireNextImageKHR(
                    ctx->device(), ctx->swapchain(), UINT64_MAX,
                    imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
                acquireSemPending_[currentFrame] = true;
                if (ctx->presentSuppressed()) pinnedSwapImage_[currentFrame] = imageIndex;
            }
            // Determinism forensics: the acquire order is presentation-engine
            // timing, i.e. wall clock — if any shading input ever depends on
            // imageIndex, this trace is how the leak shows itself (two runs
            // print different sequences at the frame the pixels diverge).
            if (const char* e = std::getenv("THREEPP_VK_TRACE_ACQUIRE"); e && *e && *e != '0') {
                std::cout << "acquire frame=" << frameSerial_ << " img=" << imageIndex << "\n";
            }
            return r;
        }

VkCommandBuffer VulkanRenderer::Impl::beginOneShot() {
            // Batch mode: every caller records into ONE shared, already-open
            // command buffer; the submit is deferred to flushOneShotBatch.
            if (oneShotBatch_) {
                if (oneShotBatchCb_ != VK_NULL_HANDLE) return oneShotBatchCb_;
            }
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(ctx->device(), &ai, &cb), "alloc one-shot cb");

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb");
            if (oneShotBatch_) oneShotBatchCb_ = cb;// keep open for later callers
            return cb;
        }

void VulkanRenderer::Impl::endAndSubmitOneShot(VkCommandBuffer cb, const char* label) {
            // Batch mode: leave the shared cb open; flushOneShotBatch submits it
            // once for the whole batch. The caller's transient resources are
            // parked (destroyBufferMaybeBatched) until the flush's wait.
            if (oneShotBatch_ && cb == oneShotBatchCb_) return;
            check(vkEndCommandBuffer(cb), "end one-shot cb");
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            const VkResult sr = vkQueueSubmit(ctx->graphicsQueue(), 1, &si, VK_NULL_HANDLE);
            if (sr != VK_SUCCESS) {
                check(sr, (std::string("submit one-shot (") + label + ")").c_str());
            }
            const VkResult wr = vkQueueWaitIdle(ctx->graphicsQueue());
            if (wr != VK_SUCCESS) {
                check(wr, (std::string("wait one-shot (") + label + ")").c_str());
            }
            vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
        }

void VulkanRenderer::Impl::flushOneShotBatch() {
            // Close the batch window. Any later one-shot returns to immediate
            // submit even if there is nothing to flush here.
            oneShotBatch_ = false;
            if (oneShotBatchCb_ == VK_NULL_HANDLE) {
                // Nothing recorded, but transient garbage may still have been
                // parked (defensive — normally empty when no cb was opened).
                for (auto& b : oneShotBatchGarbage_) destroyBuffer(ctx->allocator(), b);
                oneShotBatchGarbage_.clear();
                return;
            }
            VkCommandBuffer cb = oneShotBatchCb_;
            oneShotBatchCb_ = VK_NULL_HANDLE;
            check(vkEndCommandBuffer(cb), "end one-shot batch cb");
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            const VkResult sr = vkQueueSubmit(ctx->graphicsQueue(), 1, &si, VK_NULL_HANDLE);
            if (sr != VK_SUCCESS) check(sr, "submit one-shot batch");
            const VkResult wr = vkQueueWaitIdle(ctx->graphicsQueue());
            if (wr != VK_SUCCESS) check(wr, "wait one-shot batch");
            vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
            // The single wait above guarantees the GPU is done with every parked
            // transient (BLAS scratch, image staging) — reclaim them now.
            for (auto& b : oneShotBatchGarbage_) destroyBuffer(ctx->allocator(), b);
            oneShotBatchGarbage_.clear();
        }

void VulkanRenderer::Impl::createReservoirImages() {
            // Render extent, not swapchain extent — the per-pixel ReSTIR DI
            // reservoir images are sized to the resolution the deferred shade
            // launches at. Equal to the swapchain extent unless renderScale_ < 1.
            const VkExtent2D ext = renderExtent();
            for (size_t i = 0; i < view().reservoirPosImagesPP.size(); ++i) {
                view().reservoirPosImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "reservoirPosImagePP[0]" : "reservoirPosImagePP[1]");
            }
            for (size_t i = 0; i < view().reservoirWImagesPP.size(); ++i) {
                view().reservoirWImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                        0, i == 0 ? "reservoirWImagePP[0]" : "reservoirWImagePP[1]");
            }
            // Storage memory contents are undefined after vmaCreateImage — clear
            // both reservoir slots to 0 so the first frame's temporal-reuse read
            // sees M=0 (no prior history) instead of garbage.
            clearGbufImages();
            sampleIndex = 0;
            view().prevCameraValid = false;
            seedMotionState(lastVisibleEntries_);// forget motion history (identity next frame)
        }

void VulkanRenderer::Impl::resetAccumulation() {
            sampleIndex = 0;
            view().prevCameraValid = false;
            seedMotionState(lastVisibleEntries_);// forget motion history (identity next frame)
#if defined(THREEPP_WITH_FSR)
            fsrResetNext_ = true;// FSR treats the next dispatch as a camera cut
#endif
            if (frameState_ != FrameState::Idle) {
                pendingAccumulationReset_ = true;
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            clearGbufImages();
        }

void VulkanRenderer::Impl::scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& outResults,
                       const LidarParams& params,
                       std::vector<LidarReturn>* cleanResults) {
            const int handle = scanLidarBegin(beams, params);
            if (!scanLidarCollect(handle, outResults, cleanResults)) {
                outResults.clear();
                if (cleanResults) cleanResults->clear();
            }
        }

int VulkanRenderer::Impl::scanLidarBegin(const std::vector<LidarBeam>& beams,
                       const LidarParams& params) {
            if (beams.empty()) return vulkan::LidarScanner::kNoSlot;

            // Lazy-construct the LIDAR pipeline. Idempotent; if the user
            // never calls scanLidar, the pipeline + SBT cost is never paid.
            if (!lidar_) lidar_ = std::make_unique<vulkan::LidarScanner>(*ctx);

            // NO DEVICE DRAIN. This used to open with vkDeviceWaitIdle so the
            // TLAS and the desc buffers were quiescent; it made a 1.2 ms trace
            // cost ~28 ms, because a drain waits for every frame already queued
            // (measured: two frames in flight at ~14 ms). What the drain bought
            // is bought twice as cheaply now:
            //
            //   TLAS — the dispatch command buffer opens with a barrier whose
            //          first scope is everything submitted earlier on this
            //          queue, so an in-flight frame's build has landed before
            //          the trace reads it (see LidarScanner::dispatch).
            //   descs — read, never written, from the slot the LAST SUBMITTED
            //          frame used. The renderer flushes that slot from the
            //          frame loop under its own fence gate, so it is complete
            //          and nothing here has to write (and race with) it. The
            //          old code force-dirtied slot 0 and re-uploaded, which is
            //          exactly what needed the drain to be safe.
            //
            // render() bumps currentFrame at the end, so the just-used slot is
            // (currentFrame - 1) mod N — the same expression the fog UBO below
            // has always used, now shared by all three. Walk backwards from
            // there to the newest slot whose desc buffers actually exist: they
            // are created on their frame's first flush, so early in a session
            // (or right after a scene swap) only some slots are populated, and
            // reading an empty one would report a scene made of nothing.
            uint32_t lastSlot = (currentFrame + kFramesInFlight - 1) % kFramesInFlight;
            for (uint32_t back = 0; back < kFramesInFlight; ++back) {
                const uint32_t candidate = (currentFrame + kFramesInFlight - 1 - back) % kFramesInFlight;
                if (geometryDescsBuffers[candidate].handle != VK_NULL_HANDLE &&
                    materialDescsBuffers[candidate].handle != VK_NULL_HANDLE) {
                    lastSlot = candidate;
                    break;
                }
            }

            // Push constants encode the LIDAR equation parameters. The
            // shader multiplies the raw `laserPower · f_back · cos θ · η / r²`
            // contribution by `invReferenceIntensity` so that, AT laserPower = 1,
            // a perpendicular 1.0-albedo *Lambertian* surface at `referenceRange`
            // reads as 1.0. The reference is purely geometric (π · refRange²)
            // — it does NOT include laserPower, so raising the power slider
            // scales every return linearly.
            // The π factor absorbs the 1/π in the Lambert BRDF — without it
            // a "100% reflective" Lambertian at the reference range would
            // read as 1/π ≈ 0.318 instead of 1.0.
            vulkan_lidar::LidarPushConstants pc{};
            pc.numBeams = static_cast<uint32_t>(beams.size());
            pc.maxRange = std::max(0.001f, params.maxRange);
            pc.laserPower = std::max(0.f, params.laserPower);
            const float refRange = std::max(0.001f, params.referenceRange);
            constexpr float kPi = 3.14159265358979323846f;
            pc.invReferenceIntensity = kPi * refRange * refRange;
            pc.atmosphericExtinction = std::max(0.f, params.atmosphericExtinction);
            pc.detectorThreshold = std::max(0.f, params.detectorThreshold);
            // Frame-varying seed so the Monte Carlo fog scatter + sub-beam
            // jitter decorrelate across successive scans; sharing the path
            // tracer's accum index gives a stable, monotonically growing
            // stream that resets on accumulation reset.
            pc.rngSeed = sampleIndex;
            pc.maxReturns = std::max(1u, params.maxReturns);
            pc.samplesPerBeam = std::max(1u, params.samplesPerBeam);
            pc.beamDivergenceTan = std::tan(0.5f * std::max(0.f, params.beamDivergenceMrad) * 0.001f);
            pc.mediumSurfaceY   = params.mediumSurfaceY;
            pc.mediumExtinction = std::max(0.f, params.mediumExtinction);
            pc.mediumAlbedo     = std::clamp(params.mediumAlbedo, 0.f, 1.f);
            pc.mediumAnisotropy = std::clamp(params.mediumAnisotropy, -0.95f, 0.95f);
            pc.minRange         = std::max(0.f, params.minRange);
            pc.flags            = params.pairedCleanTrace
                                          ? vulkan_lidar::kLidarFlagPairedClean
                                          : 0u;

            // ── The ParticleField density medium (parent plan phase 3) ──────
            // Bound from exactly the source the deferred descriptor sets use,
            // so the beam and the picture cannot disagree about where the dust
            // is. Volumes are world-anchored and single-instance (not per
            // frame-in-flight), which is the same staleness contract the TLAS
            // above already has: an in-flight frame may be rewriting them, and
            // the opening barrier in LidarScanner::dispatch orders us after
            // everything already submitted.
            ensureParticleDensityResources();
            std::array<VkImageView, vulkan::kMaxDensityFields> pdViews{};
            {
                uint32_t n = 0;
                if (particleFieldPass_) {
                    for (const auto& v : particleFieldPass_->densityVolumes()) {
                        if (n >= vulkan::kMaxDensityFields) break;
                        pdViews[n++] = v.view;
                    }
                }
                for (uint32_t k = n; k < vulkan::kMaxDensityFields; ++k)
                    pdViews[k] = particleDensityDummy_.view;
            }
            vulkan::LidarScanner::DensityBinding density{};
            density.ubo       = particleDensityUbos_[lastSlot].handle;
            density.uboSize   = particleDensityUbos_[lastSlot].size;
            density.views     = pdViews.data();
            density.viewCount = vulkan::kMaxDensityFields;
            if (particleFieldPass_) {
                density.majorants     = particleFieldPass_->densityMajorants();
                density.majorantsSize = particleFieldPass_->densityMajorantsSize();
            }

            // Pack beams into the shader-side struct (vec3 + pad).
            std::vector<vulkan_lidar::LidarBeam> packed(beams.size());
            for (size_t i = 0; i < beams.size(); ++i) {
                packed[i].origin[0]    = beams[i].origin.x;
                packed[i].origin[1]    = beams[i].origin.y;
                packed[i].origin[2]    = beams[i].origin.z;
                packed[i]._pad0        = 0.f;
                packed[i].direction[0] = beams[i].direction.x;
                packed[i].direction[1] = beams[i].direction.y;
                packed[i].direction[2] = beams[i].direction.z;
                packed[i]._pad1        = 0.f;
            }

            const int handle = lidar_->dispatch(
                    ctx->graphicsQueue(),
                    tlas,
                    geometryDescsBuffers[lastSlot].handle, geometryDescsBuffers[lastSlot].size,
                    materialDescsBuffers[lastSlot].handle, materialDescsBuffers[lastSlot].size,
                    fogUbos[lastSlot].handle, fogUbos[lastSlot].size,
                    density,
                    pc,
                    packed.data(), static_cast<uint32_t>(packed.size()));
            if (handle == vulkan::LidarScanner::kNoSlot) return handle;

            // results[(beam * samplesPerBeam + sample) * maxReturns + slot],
            // and twice that many rows when the trace is paired.
            lidarRaw_[vulkan::LidarScanner::slotIndex(handle)].assign(
                    lidar_->resultSlots(handle), vulkan_lidar::LidarResult{});
            lidarPaired_[vulkan::LidarScanner::slotIndex(handle)] = params.pairedCleanTrace;
            // Freeze the entry->stable-id table this dispatch will be read
            // against. See lidarStableIds_: collect can land frames later.
            lidarStableIds_[vulkan::LidarScanner::slotIndex(handle)] = entryStableIds_;
            return handle;
        }

bool VulkanRenderer::Impl::scanLidarReady(int handle) const {
            return lidar_ && lidar_->ready(handle);
        }

bool VulkanRenderer::Impl::scanLidarCollect(int handle, std::vector<LidarReturn>& outResults,
                                            std::vector<LidarReturn>* cleanResults) {
            outResults.clear();
            if (cleanResults) cleanResults->clear();
            if (!lidar_ || handle == vulkan::LidarScanner::kNoSlot) return false;
            // Check the handle BEFORE touching the staging: a reclaimed handle
            // names a slot that now belongs to someone else, and its staging
            // holds their scan.
            if (lidar_->resultSlots(handle) == 0) return false;
            auto& raw = lidarRaw_[vulkan::LidarScanner::slotIndex(handle)];
            if (raw.empty()) return false;

            if (!lidar_->collect(handle, raw.data())) {
                raw.clear();
                return false;
            }

            // Unpack into the public LidarReturn struct. We preserve the
            // fixed-stride layout (numBeams * maxReturns) — caller filters
            // entries with hitInstanceId < 0.
            //
            // A PAIRED dispatch wrote twice that many rows, the clean leg in
            // the second half at the same stride. The degraded leg is always
            // what `outResults` gets, so a caller that ignores pairing sees
            // exactly the result it saw before pairing existed.
            const bool paired = lidarPaired_[vulkan::LidarScanner::slotIndex(handle)];
            const size_t legRows = paired ? raw.size() / 2u : raw.size();
            // The shader reports gl_InstanceCustomIndexEXT — the entry index,
            // which renumbers whenever the entry list churns. The PUBLIC
            // contract is the stable per-object id, the same number the raster
            // Ids AOV writes, so translate here: one table lookup per return,
            // no shader change, and the ray-generation / traversal / range path
            // is untouched. Sentinels (< 0: -1 miss, -2 volume scatter) are
            // passed through unmodified — they are not indices.
            const auto& ids = lidarStableIds_[vulkan::LidarScanner::slotIndex(handle)];
            auto unpack = [&ids](const vulkan_lidar::LidarResult& r, LidarReturn& o) {
                o.position.set(r.position[0], r.position[1], r.position[2]);
                o.normal.set(r.normal[0], r.normal[1], r.normal[2]);
                o.distance      = r.distance;
                o.intensity     = r.intensity;
                o.hitInstanceId = r.instanceId < 0 || size_t(r.instanceId) >= ids.size()
                                          ? r.instanceId
                                          : static_cast<int32_t>(ids[size_t(r.instanceId)]);
                o.returnNo      = r.returnNo;
                o.returnKind    = r.returnKind;
            };
            outResults.resize(legRows);
            for (size_t i = 0; i < legRows; ++i) unpack(raw[i], outResults[i]);
            if (paired && cleanResults) {
                cleanResults->resize(legRows);
                for (size_t i = 0; i < legRows; ++i) unpack(raw[legRows + i], (*cleanResults)[i]);
            }
            raw.clear();
            return true;
        }

void VulkanRenderer::Impl::reallocateRenderExtentResources() {
            // Every caller (setRenderScale, setGbufMsaa, the beginDeferredFrame
            // pending-realloc path, recreateSwapchain) has already device-idled
            // before reaching here, so the whole device is quiescent — reclaim
            // any queued retire-queue resources now rather than let them ride
            // across the realloc (they'd drain in later frames anyway, but a
            // known-idle point is the natural place, and it keeps the queue from
            // growing unbounded if the app spams resizes). Safe: device idle.
            flushRetireQueue();
            // Every image a frame-interop export copies from at this view is
            // about to be freed and rebuilt — a resize, setRenderScale,
            // setGbufferMsaa, a setSplatDepthAov toggle all funnel here. The
            // contract is to disable rather than silently reallocate under a
            // live foreign import.
            invalidateFrameInterop(view().id, "a render-target reallocation");
            for (auto& img : view().reservoirPosImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : view().reservoirWImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            createReservoirImages();// reallocates the reservoir images + clears them
            // Resize hybrid raster attachments BEFORE descriptor rewrites —
            // bindings point at rasterGbufs[f].*.view, so stale views from the
            // old extent need to be replaced before the new descriptor sets
            // capture them.
            ensureHybridResources();
            // TAA + upscale intermediates live at the render extent too;
            // rebuild before any descriptor write captures their views.
            {
                // TAA input is the render extent; history + output are the
                // swapchain extent. When they differ the resolve pass runs as
                // a temporal upsampler.
                const VkExtent2D inExt  = renderExtent();
                const VkExtent2D outExt = viewOutExtent();
                view().taa_->createImages(inExt.width, inExt.height,
                                   outExt.width, outExt.height);
            }
#if defined(THREEPP_WITH_FSR)
            // FSR stores the display (swapchain) extent at create time; a
            // renderScale change alters only the per-dispatch renderSize, so
            // recreate the context ONLY when the display extent actually changed
            // (this funnel is shared by swapchain-recreate and renderScale/MSAA).
            // The upscale output target — TaaResolve's history slot — was just
            // reallocated above. Runs BEFORE rewriteBloomDescriptors so the
            // widened HDR-plumbing gate (fsrActiveForHdrPlumbing) sees the fresh
            // state. Reset FSR history on any (re)create.
            if (fsr_) {
                const VkExtent2D disp = ctx->swapchainExtent();
                if (!fsrActive_ || fsr_->displayWidth() != disp.width ||
                    fsr_->displayHeight() != disp.height) {
                    fsrActive_    = fsr_->create(disp.width, disp.height);
                    fsrResetNext_ = true;
                }
            }
#endif
#if defined(THREEPP_WITH_DLSS)
            // Same display-extent-only recreate rule as FSR: the DLSS feature is
            // created with maxRenderSize == display, so renderScale changes are
            // covered by the per-dispatch render subrect.
            if (dlss_) {
                const VkExtent2D disp = ctx->swapchainExtent();
                if (!dlssActive_ || dlss_->displayWidth() != disp.width ||
                    dlss_->displayHeight() != disp.height) {
                    VkCommandBuffer initCb = beginOneShot();
                    dlssActive_ = dlss_->create(initCb, disp.width, disp.height,
                                                renderExtent().width,
                                                renderExtent().height);
                    endAndSubmitOneShot(initCb, "DLSS feature recreate");
                    dlssResetNext_ = true;
                    dlssHealTries_ = 0;// fresh extents → fresh self-heal budget
                }
            }
#endif
            view().bloom_->createImages(renderExtent().width, renderExtent().height);
            onAfterBloomCreateImages();
            // TAA descriptor sets are persistent (pool lives inside TaaResolve);
            // just rewrite them to the new image / view handles.
            rewriteTaaDescriptors();
            rewriteBloomDescriptors();// gbuf + TAA-input views changed (also
                                      // covers the FSR/DLSS hdrOut_ plumbing —
                                      // gated on fsrActiveForHdrPlumbing())
            rewriteDeferredDescriptors();// raster gbuf + sceneHdr views changed
        }

void VulkanRenderer::Impl::recreateSwapchainAndDescriptors() {
            ctx->recreateSwapchain();// device-idles internally
            // imageCount_ is pinned, not refreshed — see its declaration for why
            // (TaaResolve's set arrays were SIZED from it at construction, so the
            // scalar and the arrays have to move together or not at all). Nothing
            // in the negotiation can change across a recreate on this surface, but
            // "cannot happen" is worth one comparison per resize: a silently
            // changed count means every `frame * imageCount + imageIndex` in
            // TaaResolve addresses a set array that was never resized to match,
            // which is out-of-range reads at best and the wrong swapchain image at
            // worst. Fail loudly instead of rendering through it.
            const auto newImageCount = static_cast<uint32_t>(ctx->swapchainImages().size());
            if (newImageCount != imageCount_) {
                throw std::runtime_error(
                        "[VulkanRenderer] swapchain recreate changed the image count (" +
                        std::to_string(imageCount_) + " -> " + std::to_string(newImageCount) +
                        "); the TAA resolve's descriptor sets are sized for the original "
                        "count and would be indexed out of range. Rebuild the primary "
                        "view's TaaResolve here to support this.");
            }
            createRenderFinishedSemaphores();
            reallocateRenderExtentResources();
            size = WindowSize{static_cast<int>(ctx->swapchainExtent().width),
                              static_cast<int>(ctx->swapchainExtent().height)};
        }

void VulkanRenderer::Impl::setRenderScale(float scale) {
            const float clamped = scale < 0.25f ? 0.25f
                                : (scale > 1.0f ? 1.0f : scale);
            if (clamped == renderScale_) return;
            renderScale_ = clamped;
            if (frameState_ != FrameState::Idle) {
                pendingRenderScaleRealloc_ = true;
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            reallocateRenderExtentResources();
        }

void VulkanRenderer::Impl::setGbufferMsaa(uint32_t samples) {
            uint32_t clamped = samples >= 4 ? 4u : (samples >= 2 ? 2u : 1u);
            // 2x is the one count the spec doesn't guarantee (lavapipe: 1|4
            // only). Promote to 4x rather than dropping to 1x — the caller
            // asked for MSAA, and 4x is the mandatory multisample count. The
            // clamp must land in gbufMsaaSamples_ itself: the resolve/shade
            // dispatches loop over this value, so images built at one count
            // with shaders reading another would be silently wrong even where
            // it didn't crash.
            if (clamped == 2 && !gbufMsaaCountSupported(2)) {
                std::cerr << "[VulkanRenderer] gbuffer MSAA 2x is not supported by this "
                             "device (only 1x/4x are spec-guaranteed); using 4x.\n";
                clamped = 4;
            }
            if (clamped == gbufMsaaSamples_) return;
            gbufMsaaSamples_ = clamped;
            // The material sampler is 16x aniso regardless of jitter now, so
            // this mark only matters when a setTextureAnisotropy override is
            // active; kept for that path (rebuild is descriptor-cheap).
            markMaterialSamplerDirty();
            if (frameState_ != FrameState::Idle) {
                pendingRenderScaleRealloc_ = true;// shares the reallocation gate
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            reallocateRenderExtentResources();
        }

void VulkanRenderer::Impl::setSplatDepthAov(SplatDepthMode mode) {
            if (mode == splatDepthMode_) return;
            const bool wasOn = splatDepthAovAllocated();
            splatDepthMode_ = mode;
            // Only the STATISTIC changed: it rides the per-frame UBO flags, and
            // reallocating the render extent for it would stall the device for
            // a bit that costs nothing to flip.
            if (wasOn == splatDepthAovAllocated()) return;
            // ensureHybridResources only recreates the G-buffer images when the
            // EXTENT changed, and this toggle changes a FORMAT decision at the
            // same extent — so without this the AOV image stays whatever size
            // it was first allocated at (1x1, silently, for the whole run).
            // Same poke, for the same reason, as the MSAA sample-count change.
            // Primary only: a secondary view's AOV image is 1x1 either way,
            // because splats are never drawn into one.
            primaryView().rasterGbufs[0].width = 0;
            // The AOV image is allocated with the render-extent resources
            // (1x1 when off, full-res when on), so the toggle has to go back
            // through the same reallocation the render scale and the G-buffer
            // MSAA toggle use — including the mid-frame deferral, since the
            // images the in-flight command buffer names must not be freed
            // under it.
            if (frameState_ != FrameState::Idle) {
                pendingRenderScaleRealloc_ = true;// shares the reallocation gate
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            reallocateRenderExtentResources();
        }

void VulkanRenderer::Impl::beginCommandRecording(VkCommandBuffer cb) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            gpuTimings_->beginFrame(cb, currentFrame);
        }

bool VulkanRenderer::Impl::beginDeferredFrame(Object3D& scene, Camera& camera) {
            // Which projection this frame shades through. Read by the uploads
            // (parallel-ray packing, jitter placement) and by DoF, which has no
            // meaning without a lens. Stamped before the first upload below.
            view().orthoFrame_ = camera.is<OrthographicCamera>();

            VkDevice d = ctx->device();
            // ALONE in its own scope, deliberately: this is the one number that
            // says whether the CPU is the wall or is merely waiting for the GPU,
            // and folding the retire drain or the query readback into it would
            // put CPU work (or, worse, a second GPU wait) under a label that is
            // read as "GPU back-pressure".
            {
                THREEPP_CPUPROF("frame.0_fenceWait");
                vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);
            }
            // Fence signaled ⇒ frame (frameSerial_ - kFramesInFlight) is
            // GPU-complete (this slot's previous occupant). Reclaim every
            // resource retired by that frame or earlier before touching this
            // slot's buffers/descriptors. See VulkanRetireQueue.hpp.
            {
                THREEPP_CPUPROF("frame.0a_retireDrain");
                drainRetireQueue();
            }
            // Fence has signaled → the previous render that wrote into this
            // frame's query pool has retired. Read it now, before we reset
            // the pool and re-record. Result is stored in gpuTimings_ for
            // the public getter to read.
            // Its own phase, not part of the fence wait: every pair is fetched
            // with VK_QUERY_RESULT_WAIT_BIT (GpuTimings.cpp), so "returns
            // immediately because the fence already signaled" is an assumption,
            // and a separate key is what makes it checkable.
            {
                THREEPP_CPUPROF("frame.0b_queryReadback");
                gpuTimings_->readBack(currentFrame, pendingCpuEnsureSceneMs_);
            }
            // Same fence, same reasoning, for the DisplacedMesh (FFT water)
            // CPU height mirrors. Each ocean update ensureSceneBuilt queued
            // this frame copies its height field into readback slot
            // currentFrame once recorded below; the slot's PREVIOUS occupant
            // (kFramesInFlight frames ago) is what the fence just retired, so
            // its copy is whole — mirror it now, before this frame's record
            // reuses the slot. Taken at stage time (ensureSceneBuilt, before
            // this wait) the memcpy raced the in-flight frames' copies, and
            // sampleHeight() returned a field whose age and tearing depended
            // on host pacing; this is a fixed kFramesInFlight-frame latency.
            for (auto& [dmPtr, stPtr, tsec] : pendingDisplacedDeforms_) {
                (void)tsec;
                mirrorDisplacedHeightfields(*dmPtr, *stPtr);
            }

            // Apply any setter requests deferred from mid-frame. setRenderScale
            // / resetAccumulation issue vkDeviceWaitIdle + reallocate descriptor
            // pool / clear images — both unsafe with an open cmd buffer in the
            // prior frame. We're past the fence wait now so the GPU is idle for
            // *this* slot at minimum; vkDeviceWaitIdle below drains the rest.
            if (pendingRenderScaleRealloc_ || pendingAccumulationReset_ || pendingViewChanges_) {
                vkDeviceWaitIdle(d);
                if (pendingRenderScaleRealloc_) {
                    reallocateRenderExtentResources();// also rewrites deferred descriptors
                    pendingRenderScaleRealloc_ = false;
                }
                if (pendingAccumulationReset_) {
                    // Every live view: the reset is a per-view act (each view
                    // owns its histories) and the device is idle here.
                    forEachLiveView([&] { clearGbufImages(); });
                    pendingAccumulationReset_ = false;
                }
                // Views added / removed since the last frame. Same reasoning as
                // the two above: the caller asked mid-frame, we act here.
                if (pendingViewChanges_) applyPendingViewChanges();
            }

            uint32_t imageIndex = 0;
            // Third place the CPU can block on something that is not the CPU:
            // with only kFramesInFlight swapchain images this waits for the
            // presentation engine to release one. Own key, so a compositor stall
            // cannot be read as CPU work — or as fence-wait back-pressure.
            // (Free after the first frame per slot on a suppressed-present
            // swapchain — there is no presentation engine in that picture.)
            VkResult acq;
            {
                THREEPP_CPUPROF("frame.0c_acquire");
                acq = acquireOrReuseSwapchainImage(imageIndex);
            }
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return false;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }
            frameImageIndex_ = imageIndex;

            {
                THREEPP_CPUPROF("frame.1_uboUpdates");
                updateCameraUbo(currentFrame, camera);
                updateLightsUbo(currentFrame, scene);
                updateFogUbo(currentFrame, scene, camera);
                updateCloudUbo(currentFrame);
            }
            // Safe to write motionMatBuffers[currentFrame] now that the
            // inFlight[currentFrame] fence has been signaled — the GPU has
            // finished its previous use of this slot.
            computeAndUploadMotionMatrices(currentFrame, lastVisibleEntries_);
            // Same fence guarantee covers materialDescsBuffers[currentFrame].
            // ensureSceneBuilt staged any material-value change in
            // matDescsCached_ and marked the entry ranges it patched; flush this
            // slot's ranges now (the other slot flushes when its frame comes
            // around, which is why the pending set is PER SLOT).
            flushMaterialDescsIfDirty(currentFrame);
            // Stamp per-entry MOVED state (meshMovedBits_, finalized in
            // ensureSceneBuilt) into GeometryDesc._pad so the reflection/GI ray-hit
            // handler can tell when it reflects a MOVING object and reset its
            // temporal history, and the reproject guards can reject history
            // inherited from a moving mesh (a moving reflected object can't be
            // temporally integrated — the surface reproject tracks the SURFACE,
            // not the moving content, so it smears into a ghost trail).
            // STICKY: "moved" holds for kMovedStickyFrames after the last actual
            // change. A fixed-substep integrator under variable dt produces render
            // frames with ZERO substeps — the driven car's transform stalls for a
            // frame, the per-frame bit drops, and every _pad-gated guard goes
            // inert for exactly that frame: the car's reflection slips into the
            // vacated road's history and fades out as an intermittent ghost
            // AFTERIMAGE of the car ("blinks" under engine jank, never on a
            // smooth gravity roll). Only re-uploads on a 0↔1 transition (the
            // countdown is host-side), so a settled scene still pays nothing.
            {
                THREEPP_CPUPROF("frame.B_movedSticky");
                constexpr uint32_t kMovedStickyFrames = 30;
                if (meshMovedSticky_.size() < geomDescsCached_.size())
                    meshMovedSticky_.resize(geomDescsCached_.size(), 0u);
                // Fast skip: with no moved bits set this frame AND no live
                // countdowns, every sticky value stays 0 and every flag bit
                // stays as-is — the walk is a no-op. Settled scenes (the
                // common case) pay one word-scan instead of O(entries).
                bool anyMoved = false;
                for (const uint32_t w : meshMovedBits_)
                    if (w != 0u) { anyMoved = true; break; }
                if (anyMoved || stickyActiveCount_ > 0u) {
                    bool changed = false;
                    uint32_t active = 0;
                    for (size_t i = 0; i < geomDescsCached_.size(); ++i) {
                        const uint32_t w = static_cast<uint32_t>(i) >> 5u;
                        const uint32_t bit = 1u << (static_cast<uint32_t>(i) & 31u);
                        const bool movedNow = w < meshMovedBits_.size() && (meshMovedBits_[w] & bit);
                        if (movedNow) meshMovedSticky_[i] = kMovedStickyFrames;
                        else if (meshMovedSticky_[i] > 0u) --meshMovedSticky_[i];
                        const uint32_t moved = meshMovedSticky_[i] > 0u ? 1u : 0u;
                        if (moved) ++active;
                        // Bit 0 only — bits 1..3 carry the packed-attribute mask
                        // (GeometryDesc::flags) and must survive the stamp.
                        const uint32_t nf = (geomDescsCached_[i].flags & ~1u) | moved;
                        if (geomDescsCached_[i].flags != nf) {
                            geomDescsCached_[i].flags = nf;
                            // Range marked at the write, ascending, so a cohort of
                            // grains that all start (or all settle) together is one
                            // range rather than one whole-array resend. A scattered
                            // enough set promotes itself back to whole-array inside
                            // DescDirtyRanges::mark.
                            markGeomDescsDirty(static_cast<uint32_t>(i));
                            changed = true;
                        }
                    }
                    stickyActiveCount_ = active;
                    if (changed) {
                        ++drawInputsVersion_;// kInstFlagMoving transitions reshape DrawInfo
                    }
                }
            }
            // Same fence guarantee covers geometryDescsBuffers[currentFrame]:
            // an auto-LOD level switch patched geomDescsCached_ + marked
            // geomDescsDirty_ in ensureSceneBuilt; landing the flush here —
            // post-fence, pre-record — keeps this frame's GeometryDescs
            // consistent with this frame's TLAS (which references the newly
            // selected level BLASes) without any device stall.
            flushGeometryDescsIfDirty(currentFrame);
            // Per-frame frustum cull: tags every entry with `inFrustum`
            // for the raster passes to consume.
            cullEntriesAgainstFrustum(camera);
            // Hybrid raster prepass: lazy-create resources on first use,
            // refresh attachments on resize, then upload the per-frame
            // camera VPs (curr jittered, curr unjittered, prev unjittered).
            // Must run after computeAndUploadMotionMatrices so the descriptor
            // rewrite picks up the populated motionMat buffer for this frame.
            {
                THREEPP_CPUPROF("frame.2_ensureHybridRes");
                ensureHybridResources();
                uploadRasterCameraUbo(currentFrame, camera);
            }
            // Build the per-frame DrawInfo + indirect-cmd buffers used
            // by the indirect-drawing gbuf pass. Runs after the cull
            // pass + camera upload (depends on both) and before record.
            buildIndirectDrawData(currentFrame);
            // GPU per-instance world matrices — the producer half only; nothing
            // above or below reads its output yet (stage 1 of
            // plans/gpu-driven-instances.md). Placed after ensureHybridResources
            // because that is where instExpand_ is created, and inside the
            // post-fence window because it writes this slot's host-mapped pools
            // and this slot's descriptor set.
            prepareInstanceExpansion(currentFrame);
            // ParticleField position rings + FieldDesc SSBO. Same site for the
            // same reason (R6): this writes this slot's host-visible buffers,
            // and phase 1's descriptor write lands here too.
            prepareParticleFields(currentFrame);
            // Splat reflection volumes (plans/splat-volume-reflections.md).
            // Beside prepareParticleFields because it is the same kind of work
            // for the same reason (R6): it writes THIS slot's host-visible UBO
            // and can rewrite this slot's descriptor set, and both are legal
            // only inside the post-fence window.
            //
            // The plan names collectSplatClouds' syncClouds call as the site,
            // for the ordering it guarantees — a cloud uploaded this frame is
            // BAKED before the UBO names it. That ordering still holds here
            // (collectSplatClouds runs before renderFrame in the same render()
            // call), and this site adds the guarantee that one cannot give:
            // syncClouds runs BEFORE the inFlight[currentFrame] fence wait, so
            // writing this slot's UBO or descriptor set there would touch
            // resources the GPU may still be reading.
            updateSplatVolumeUbo(currentFrame);
            // F4: the billboard glow chain, which prepareParticleFields has just
            // told us whether any field wants. Created here rather than during
            // recording so the one-off pipeline compile and image allocation
            // land in the prepare window like every other lazy resource.
            ensureFieldBillboardGlow();
            {
                THREEPP_CPUPROF("frame.I3_uploadMovedBits");
                uploadMeshMovedBits(currentFrame);
            }
            // Same fence guarantees emissiveTriBuffers[currentFrame] is no
            // longer in use; rebuild the per-frame CDF and rewrite the emissive
            // binding if the buffer grew.
            if (buildAndUploadEmissiveTris(currentFrame, lastVisibleEntries_)) {
                // Keep the deferred pass's emissive binding fresh when the
                // per-frame buffer grows. EVERY view: growth destroyed the old
                // buffer outright, and recordSecondaryViews records against
                // this same slot later this frame — a secondary's set left
                // stale names freed memory (an intermittent device-lost once
                // VMA recycles the block, not a visual glitch). Safe to
                // rewrite here: the slot's fence has signaled, and the
                // secondaries ride the same submission.
                THREEPP_CPUPROF("frame.2c_descRefresh");
                forEachLiveView([&] {
                    if (view().deferredShade_) {
                        view().deferredShade_->rewriteEmissive(currentFrame,
                                                        emissiveTriBuffers[currentFrame].handle);
                    }
                });
                if (primaryView().deferredShade_) {
                    probeGI_->rewriteEmissive(currentFrame,
                                              emissiveTriBuffers[currentFrame].handle);
                }
            }
            // Same key as above: the registry sums a name across call sites, and
            // both of these are "a descriptor/env refresh landed in this frame".
            // The env branch drains the device and clears the G-buffer, so a
            // single occurrence is one enormous outlier — that is exactly what a
            // windowed mean is for.
            if (refreshEnvTextureFromScene(scene)) {
                THREEPP_CPUPROF("frame.2c_descRefresh");
                // Env is a primary radiance source — can't reproject, so this
                // path already wipes the ReSTIR DI reservoir history (a
                // vkDeviceWaitIdle + clearGbufImages) to cold-start next frame.
                // Drain FIRST so the all-slots descriptor rewrite lands while the
                // device is idle (both FIF sets safe to touch), and flush the
                // retire queue (refreshEnvTextureFromScene retired the old env
                // images) now that nothing references them.
                vkDeviceWaitIdle(ctx->device());
                flushRetireQueue();
                // EVERY view. The old env images were just retired and freed;
                // a secondary whose descriptor still names them samples freed
                // memory, which reads as a black sky rather than as a crash.
                forEachLiveView([&] {
                    rewriteDeferredDescriptors();
                    clearGbufImages();
                });
                // The splat sets too: collectSplatClouds ran BEFORE this frame
                // began and wrote them against the env image retired above —
                // on the very first frame that is the default 1x1 env, freed
                // by the flush while this frame's splat dispatch is still to
                // be recorded. setEnvironment's dirty flag only rewrites at
                // the NEXT syncClouds, one frame too late.
                if (splat_)
                    splat_->rewriteEnvironment(envImage.view, envImage.sampler,
                                               envImage.mipLevels);
            }

            // Per-FIF deferred-descriptor refresh: a material texture swapped in
            // place (refreshDirtyMaterialTextures) marked all FIF sets dirty.
            // This slot's fence signaled at the top of the frame, so rewrite ONLY
            // its set now — the other slot refreshes when its frame comes around.
            // No-op if the env path above already did an all-slots rewrite.
            if (deferredDescDirty_[currentFrame]) {
                THREEPP_CPUPROF("frame.2c_descRefresh");
                // Per-view sets, one scene-wide cause: the swapped texture is
                // in the bindless array every view's set points at.
                forEachLiveView([&] {
                    rewriteDeferredDescriptors(static_cast<int>(currentFrame));
                });
            }

            vkResetFences(d, 1, &inFlight[currentFrame]);
            // Per-frame hook: CPU readbacks (e.g. auto-exposure histogram) that are
            // safe only after the current slot's prior GPU work is retired (fence above).
            {
                THREEPP_CPUPROF("frame.3_onBeginHook");
                const double now = frameNowSec();
                const float  dt  = (lastFrameTime_ > 0.0)
                                   ? static_cast<float>(now - lastFrameTime_) : 0.016f;
                lastFrameTime_ = now;
                onBeginDeferredFrame(currentFrame, dt);
            }
            {
                THREEPP_CPUPROF("frame.3a_cbBegin");
                vkResetCommandBuffer(cmdBuffers[currentFrame], 0);
                beginCommandRecording(cmdBuffers[currentFrame]);
            }
            // First thing in the stream, and its own phase: one dispatch, no
            // dependants. Deliberately NOT inside frame.J_record — a phase whose
            // whole purpose is a cost measurement must not be summed into
            // another one.
            {
                THREEPP_CPUPROF("frame.M3_instExpandRec");
                recordInstanceExpansion(cmdBuffers[currentFrame], currentFrame);
                // The ParticleField device emitter, FIRST of the field block:
                // one dispatch per Ownership::Renderer field writes this
                // frame's positions AND its prevPositions, and everything below
                // — the density scatter here, every view's G-buffer draw later
                // — reads them. Once for all views, same world-anchored
                // argument as the scatter.
                recordParticleFieldEmit(cmdBuffers[currentFrame], currentFrame);
                // Same place, same shape: a handful of 4-byte copies that give
                // each ParticleField's draw command its instanceCount without
                // the count ever being a CPU-visible value.
                recordParticleFieldCounts(cmdBuffers[currentFrame]);
                // And the density representation's whole per-frame GPU cost:
                // clear + splat each dust field into its world-anchored volume.
                // HERE, not per view: the volume is world-anchored precisely so
                // K cameras share ONE scatter (plan R9), and it must precede
                // every view's froxel pass, all of which record later into this
                // same command buffer.
                recordParticleDensityScatter(cmdBuffers[currentFrame], currentFrame);
            }
            // Record the full deferred-render body into the now-open cmd
            // buffer. Leaves the swapchain image in GENERAL.
            // INCLUSIVE of frame.L_tlasRefitDispatch (and therefore of
            // frame.H_uploadTlasInst inside it), which recordDeformAndTlas
            // reaches. Those two are reported as detail rows; only ONE of
            // {frame.J_record} / {L, H} belongs in a phase sum.
            {
                THREEPP_CPUPROF("frame.J_record");
                recordCommandBuffer(cmdBuffers[currentFrame], imageIndex);
            }

            // Every secondary view, into the SAME command buffer and therefore
            // the same submission. They share this frame's TLAS/BLAS, lights,
            // materials and textures — only the genuinely per-camera work is
            // repeated. Placed after the primary so the shared scene is fully
            // built and its barriers are already in the stream, and before the
            // capture / event-camera / overlay tail, all of which are the
            // primary's alone.
            // NOTE, for anyone reading frame.C_frustumCull / frame.D_buildIndirect
            // on a multi-view scene: this re-enters both, so those two keys ALREADY
            // sum primary + every secondary under one name. Single-view scenes
            // return immediately here and are unaffected.
            {
                THREEPP_CPUPROF("frame.J9_secondaryViews");
                recordSecondaryViews(cmdBuffers[currentFrame]);
            }

            // Scene-only swapchain capture. Runs BEFORE the sprite + ImGui
            // overlays composite — sensor pipelines that consume the
            // renderer's output need a clean image without their own
            // visualisation drawn on top of it, otherwise they'd see
            // their own readout as scene motion and feedback-loop.
            // One key for the whole record tail (capture, event camera, view
            // composite, screen-space sprite overlay): individually these are
            // early-outs on a scene that uses none of them, and the point of the
            // key is that the books close, not that each is attributable.
            THREEPP_CPUPROF("frame.4_recordTail");
            if (sceneCaptureEnabled_) {
                recordSceneCapture(cmdBuffers[currentFrame], imageIndex);
            }

            // Zero-copy frames out (enableFrameInterop). The SAME point in the
            // frame as the scene capture, and for the same reason: the Color
            // channel must be the clean post-TAA picture, without the sprite /
            // ImGui overlays or a picture-in-picture view composite drawn into
            // it. The G-buffer AOVs are equally final by here — every consumer
            // of them (deferred shade, TAA, the secondary views) has already
            // been recorded. Records nothing when no view is armed.
            recordFrameInterop(cmdBuffers[currentFrame], imageIndex);

            // GPU event camera detection. Run event_shade first to fill
            // eventLumaBuf_ — with deterministic Lambert lighting from the
            // gbuf (Shaded source: no stochastic shading noise as a source of
            // false events) or with a box-average of the final frame now
            // sitting in the swapchain (Final source: everything the picture
            // shows fires) — then dispatch the detector against that buffer.
            if (eventCamEnabled_ && eventCam_ &&
                eventShadePipeline_ != VK_NULL_HANDLE) {
                recordEventShade(cmdBuffers[currentFrame], currentFrame, imageIndex);
                eventCam_->record(cmdBuffers[currentFrame],
                                  eventLumaBuf_.handle,
                                  eventCamParams_);
            }

            // Any secondary view the caller asked to see, copied into the
            // frame at its rect. After the scene capture deliberately: a
            // sensor consuming that capture wants the primary camera's image,
            // not a picture-in-picture of some other camera. Before the
            // overlay, so ImGui and screen-space sprites still draw over it.
            recordViewComposite(cmdBuffers[currentFrame], imageIndex);

            // Screen-space sprite auto-overlay. Walks the main scene for
            // Sprites with screenSpace=true and composites them through
            // an internal ortho camera derived from the swapchain extent —
            // no user code beyond setting the flag on the sprite.
            // recordOrthoOverlay short-circuits when no eligible sprites
            // are found, so the typical no-sprite case pays only the walk.
            const VkExtent2D extPT = ctx->swapchainExtent();
            if (!screenSpaceCam_) {
                screenSpaceCam_ = OrthographicCamera::create(
                        0.f, float(extPT.width), float(extPT.height), 0.f,
                        0.1f, 10.f);
                screenSpaceCam_->position.z = 1.f;
            } else {
                screenSpaceCam_->left   = 0.f;
                screenSpaceCam_->right  = float(extPT.width);
                screenSpaceCam_->top    = float(extPT.height);
                screenSpaceCam_->bottom = 0.f;
                screenSpaceCam_->updateProjectionMatrix();
            }
            overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                 scene, *screenSpaceCam_, /*screenSpaceOnly=*/true);
            return true;
        }

bool VulkanRenderer::Impl::beginFrameOrthoOnly() {
            VkDevice d = ctx->device();
            // DISTINCT names from the deferred path's, even though it is the same
            // fence ring and the same swapchain: the registry keys on the string,
            // so reusing frame.0_fenceWait here would silently merge two code
            // paths into one number.
            {
                THREEPP_CPUPROF("frame.0y_fenceWaitOrtho");
                vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);
            }
            // Same retire reclaim as beginDeferredFrame (the ortho-only frame
            // path shares the fence ring). See VulkanRetireQueue.hpp.
            drainRetireQueue();
            gpuTimings_->readBack(currentFrame, pendingCpuEnsureSceneMs_);

            uint32_t imageIndex = 0;
            VkResult acq;
            {
                THREEPP_CPUPROF("frame.0z_acquireOrtho");
                acq = acquireOrReuseSwapchainImage(imageIndex);
            }
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return false;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }
            frameImageIndex_ = imageIndex;

            vkResetFences(d, 1, &inFlight[currentFrame]);
            vkResetCommandBuffer(cmdBuffers[currentFrame], 0);
            VkCommandBuffer cb = cmdBuffers[currentFrame];
            beginCommandRecording(cb);

            // UNDEFINED → TRANSFER_DST, vkCmdClearColorImage, TRANSFER_DST → GENERAL.
            const VkImage img = ctx->swapchainImages()[imageIndex];
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            {
                VkImageMemoryBarrier2 b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                b.srcAccessMask = 0;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img;
                b.subresourceRange = range;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cb, &dep);
            }
            // Encode into the swapchain's display (sRGB) space. The swapchain is a
            // plain UNORM image and this clear bypasses the shader's linearToSRGB,
            // so a color-managed (linear) clear color must be encoded here to match
            // shaded pixels. No-op when ColorManagement is disabled (legacy raw).
            Color cc;
            cc.copy(clearColor);
            ColorManagement::workingToColorSpace(cc, SRGBColorSpace);
            VkClearColorValue cv{};
            cv.float32[0] = cc.r;
            cv.float32[1] = cc.g;
            cv.float32[2] = cc.b;
            cv.float32[3] = clearAlpha;
            vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &cv, 1, &range);
            {
                VkImageMemoryBarrier2 b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                  VK_ACCESS_2_TRANSFER_READ_BIT |
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                  VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img;
                b.subresourceRange = range;
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cb, &dep);
            }
            return true;
        }

void VulkanRenderer::Impl::endFrame() {
            if (frameState_ == FrameState::Idle) return;

            const uint32_t imageIndex = frameImageIndex_;
            VkCommandBuffer cb = cmdBuffers[currentFrame];

            // EVERY frame.K* phase is OUTSIDE cpuFrameMs: endFrame() runs from the
            // Canvas frame-end callback (see the constructor), i.e. after render()
            // has already returned and already stamped cpuFrameMs. And because
            // cpuprof::Registry::endFrame() fires at the tail of render(), these
            // samples land in the NEXT window — fine for a windowed mean, wrong for
            // reconciling one frame.
            {
                THREEPP_CPUPROF("frame.K1_endRecord");
                recordOverlayAndPresentTransition(cb, imageIndex);
                // Closes the whole-command-buffer GPU bracket opened in
                // GpuTimings::beginFrame; must be the last recorded command.
                gpuTimings_->endFrameTotal(cb, currentFrame);
                check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
                gpuTimings_->finishRecord();
            }

            VkSemaphoreSubmitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfo.semaphore = imageAvailable[currentFrame];
            // Must cover EVERY stage that first touches the swapchain image. The
            // presentation engine's release is only ordered against the stages
            // named here, and these stage chains are independent of each other —
            // waiting at RAY_TRACING alone (as this used to) ordered nothing,
            // because raygen no longer writes the swapchain: the first touches are
            // the UNDEFINED->GENERAL transition (COMPUTE|TRANSFER dst), the clear /
            // blit paths (TRANSFER, which subsumes CLEAR/COPY/BLIT), the TAA and
            // upscale stores (COMPUTE) and the overlay/ImGui dynamic-rendering
            // passes (COLOR_ATTACHMENT_OUTPUT). Deliberately not ALL_COMMANDS, so
            // the frame's offscreen work still overlaps the acquire.
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

            // Per-IMAGE, not per-frame: safe to re-signal only once this image
            // index has been re-acquired, which is exactly when the prior
            // present's wait on it is guaranteed consumed (see the member note).
            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = renderFinished[imageIndex];
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            VkCommandBufferSubmitInfo cbInfo{};
            cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            cbInfo.commandBuffer = cb;

            // Suppressed presents (headless canvas) change BOTH semaphores, and
            // for spec reasons rather than tidiness:
            //  - imageAvailable is a BINARY semaphore signalled by the acquire.
            //    Each acquire signals it once, so only the frame that follows an
            //    acquire may wait on it; a second wait would never be satisfied.
            //    On a pinned slot that is the slot's FIRST frame only.
            //  - renderFinished exists solely for the present to wait on. With no
            //    present nothing ever waits, so signalling it again next frame
            //    would signal an already-signalled binary semaphore
            //    (VUID-vkQueueSubmit2-semaphore-03868). Drop the signal instead.
            // The inFlight fence is unchanged and remains what orders the slot's
            // reuse of the image and of every per-frame buffer.
            const bool suppressPresent = ctx->presentSuppressed();
            VkSubmitInfo2 submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submit.waitSemaphoreInfoCount = acquireSemPending_[currentFrame] ? 1u : 0u;
            submit.pWaitSemaphoreInfos = &waitInfo;
            submit.commandBufferInfoCount = 1;
            submit.pCommandBufferInfos = &cbInfo;
            submit.signalSemaphoreInfoCount = suppressPresent ? 0u : 1u;
            submit.pSignalSemaphoreInfos = &signalInfo;
            // Separate from present, deliberately: a submit that stalls means
            // driver-side command translation, a present that stalls means the
            // compositor. Different diagnoses, so different keys.
            {
                THREEPP_CPUPROF("frame.K2_submit");
                check(vkQueueSubmit2(ctx->graphicsQueue(), 1, &submit, inFlight[currentFrame]),
                      "vkQueueSubmit2");
            }
            acquireSemPending_[currentFrame] = false;// consumed by the submit above

            if (suppressPresent) {
                // Nothing to present to. The image stays app-owned in PRESENT_SRC
                // (recordOverlayAndPresentTransition left it there, which is what
                // readRGBPixels expects), and this slot re-renders into the same
                // image next time round. Resize is the one thing that still has
                // to be honoured — normally it rides in on the present's result.
                if (needsResize) {
                    needsResize = false;
                    recreateSwapchainAndDescriptors();
                }
            } else {
                VkPresentInfoKHR pi{};
                pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
                pi.waitSemaphoreCount = 1;
                pi.pWaitSemaphores = &renderFinished[imageIndex];
                VkSwapchainKHR sc = ctx->swapchain();
                pi.swapchainCount = 1;
                pi.pSwapchains = &sc;
                pi.pImageIndices = &imageIndex;

                // Brace the present ALONE — the recreate path below is a swapchain
                // rebuild, not a present, and must not wear this label. With vsync off
                // this should be near zero; if it is not, the present mode is a
                // confound for every fps number measured through this renderer.
                VkResult pr;
                {
                    THREEPP_CPUPROF("frame.K3_present");
                    pr = vkQueuePresentKHR(ctx->presentQueue(), &pi);
                }
                if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || needsResize) {
                    needsResize = false;
                    recreateSwapchainAndDescriptors();
                } else if (pr != VK_SUCCESS) {
                    check(pr, "vkQueuePresentKHR");
                }
            }

            // Cap to keep `subIdx = sampleIndex * spp + s` (deferred_shade.comp)
            // from overflowing uint32. With spp ≤ 256, cap at 2^24 leaves headroom
            // (16M·256 ≈ 4G, just under uint32 max). The previous 65535 cap froze
            // the blue-noise jitter and Halton sequence after ~18 min at 60 fps,
            // causing the per-pixel FC=4096 running mean to slowly absorb the
            // single deterministic sample — image visibly resets from converged
            // toward biased noise. 16M ≈ 75 hours at 60 fps, no longer reachable.
            if (sampleIndex < (1u << 24)) ++sampleIndex;

            // Advance serial + slot together, only on a SUBMITTED frame. A
            // failed acquire returns before endFrame(), so serial never runs
            // ahead of slot — keeping the retire queue's serial↔slot fence
            // invariant exact (VulkanRetireQueue.hpp).
            ++frameSerial_;
            // Same cadence for the 3D-overlay line cache's lastTouch clock, so
            // its stale sweep has a monotonic reference. It previously sat at 0
            // forever, which made every entry look freshly touched and meant the
            // cache only ever grew.
            ++overlayFrameCounter_;
            sweepLineGeomCache();
            currentFrame  = (currentFrame + 1) % kFramesInFlight;
            frameState_   = FrameState::Idle;
        }

void VulkanRenderer::Impl::renderFrame(Object3D& scene, Camera& camera) {
            // "Ortho" here means "route to the 2D overlay path", not "the camera
            // is an OrthographicCamera". With setOrthographicSceneRendering on,
            // a standalone ortho render is a 3D view and belongs on the deferred
            // path; the HUD pattern's second call (a frame already in flight)
            // is unaffected either way — orthoSceneRender() only answers true
            // from Idle.
            const bool isOrtho = camera.is<OrthographicCamera>() && !orthoSceneRender(camera);

            if (frameState_ == FrameState::Idle) {
                if (isOrtho) {
                    if (!beginFrameOrthoOnly()) return;
                    frameState_ = FrameState::RecordingOrthoOnly;
                    // Standalone 2D render: draw the ortho overlay (Sprites +
                    // Line/LineSegments) now, so a single render(scene, orthoCam)
                    // call produces output. The HUD pattern (a perspective render
                    // first, then ortho) instead reaches recordOrthoOverlay via the
                    // frame-already-in-flight path below.
                    overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                         scene, camera, /*screenSpaceOnly=*/false);
                } else {
#if defined(THREEPP_WITH_DLSS)
                    // Self-heal a sticky NGX evaluate failure (0xBAD00005 after
                    // an extent transition the resize funnel didn't observe):
                    // do automatically what a manual window resize does —
                    // recreate the feature at the CURRENT extents. Rare path,
                    // so the idle is acceptable; bounded tries so a persistent
                    // failure degrades to the FSR/TAA fallback instead of a
                    // recreate/hitch loop (the resize funnel resets the budget).
                    if (dlss_ && dlssActive_ && dlss_->failing()) {
                        if (dlssHealTries_ < 3) {
                            ++dlssHealTries_;
                            vkDeviceWaitIdle(ctx->device());
                            const VkExtent2D disp = ctx->swapchainExtent();
                            VkCommandBuffer initCb = beginOneShot();
                            dlssActive_ = dlss_->create(initCb, disp.width, disp.height,
                                                        renderExtent().width,
                                                        renderExtent().height);
                            endAndSubmitOneShot(initCb, "DLSS feature self-heal recreate");
                            dlssResetNext_ = true;
                            if (view().taa_) view().taa_->invalidateHistory();
                        } else {
                            dlssActive_ = false;// give up until the next resize
                            std::fprintf(stderr,
                                         "[threepp] DLSS: persistent evaluate failure - "
                                         "disabled until the next display resize (FSR/TAA fallback).\n");
                        }
                    }
#endif
                    if (!beginDeferredFrame(scene, camera)) return;
                    frameState_ = FrameState::RecordingPostShade;
                }
                return;
            }

            // Frame already in flight from a prior render() call this iteration.
            if (!isOrtho) {
                // Split-screen secondary pane: when a scissor sub-rect is set, a
                // second perspective render() composes THIS scene+camera into
                // that region of the open frame, beside the primary
                // deferred-render pane — without touching the accumulation/TLAS
                // state of that pane. The pane is a LIT PANE (OverlayPass): rect
                // cleared to the scene background, meshes depth-tested and
                // sun+ambient shaded, Points/Lines/Sprites on top. A preview,
                // not the deferred pipeline — the editor's camera dock and
                // multiple_scenes' second view read as the same scene under the
                // same sun, but shadows/GI/fog stay a multi-view feature.
                // Without a scissor, fall back to the old behavior (finalize
                // the prior frame, restart for this one).
                if (scissorTest && scissor.z >= 1.f && scissor.w >= 1.f) {
                    const VkExtent2D full = ctx->swapchainExtent();
                    const uint32_t rx = static_cast<uint32_t>(std::clamp(static_cast<int>(scissor.x), 0, static_cast<int>(full.width)));
                    const uint32_t rw = static_cast<uint32_t>(std::clamp(static_cast<int>(scissor.z), 1, static_cast<int>(full.width) - static_cast<int>(rx)));
                    const uint32_t rh = static_cast<uint32_t>(std::clamp(static_cast<int>(scissor.w), 1, static_cast<int>(full.height)));
                    const int      syB = std::clamp(static_cast<int>(scissor.y), 0, static_cast<int>(full.height) - static_cast<int>(rh));
                    const uint32_t ry = static_cast<uint32_t>(static_cast<int>(full.height) - (syB + static_cast<int>(rh)));
                    // Hand the pane the scene's equirect environment so its sky
                    // matches the deferred primary miss. Only a REAL texture:
                    // a background colour (or no env at all) keeps the pane's
                    // verbatim colour clear — the deferred solid-bg bypass
                    // shows that colour display-referred and tone-mapping it
                    // in the pane would make the two views disagree.
                    overlayPass_->setPaneEnvironment(
                            (!envIsDefault && !envIsBgColor) ? envImage.view : VK_NULL_HANDLE,
                            envImage.sampler, currentExposure());
                    overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                         scene, camera, /*screenSpaceOnly=*/false, rx, ry, rw, rh);
                    return;
                }
                // User issued a second full-frame perspective render — finalize
                // the prior frame, then re-enter from Idle for the new one.
                endFrame();
                renderFrame(scene, camera);
                return;
            }
            // HUD-style ortho render: append Sprite draws onto the open
            // cmd buffer's swapchain image. Mesh / Line HUD overlays are
            // a follow-up — the no-op branch falls through to here so the
            // present still completes from endFrame().
            overlayPass_->record(cmdBuffers[currentFrame], currentFrame, frameImageIndex_,
                                 scene, camera, /*screenSpaceOnly=*/false);
        }

// ── Multi-view: lifecycle ───────────────────────────────────────────────────
//
// A view is a persistent object. Everything it needs is allocated once, here,
// behind a device drain; from then on rendering it is just a second pass over
// the frame's already-built scene. Nothing in the per-frame path adds, removes
// or resizes a view — per-frame churn of GPU-visible lists means a rebuild +
// vkDeviceWaitIdle every frame, which persistent views exist to avoid.

VulkanRenderer::Impl::ViewContext* VulkanRenderer::Impl::findView(uint32_t handle) {
            for (auto& v : views_) {
                if (v->id != handle) continue;
                // A view awaiting its deferred free is already GONE as far as
                // the public API is concerned. The caller asked for it to be
                // removed; that its memory is released at the next frame
                // boundary is an implementation detail, and letting a readback
                // through in the meantime would hand back labels for a camera
                // the caller believes no longer exists.
                return v->pendingDestroy ? nullptr : v.get();
            }
            return nullptr;
        }

void VulkanRenderer::Impl::createSecondaryViewResources(ViewContext& v) {
            // Runs with curView_ pointed at `v`, so every createXxx / rewriteXxx
            // below — all of which were made view-relative in the ViewContext
            // extraction — allocates into this view without knowing it exists.
            // That is the payoff of the extraction: the setup sequence here is
            // the same one the primary runs in the constructor.
            ViewContext* saved = curView_;
            curView_ = &v;

            const VkDeviceSize before = vmaAllocatedBytes();

            // Colour target: the "swapchain of one" this view's TaaResolve
            // resolves into. Swapchain FORMAT so the entire post/TAA tail
            // writes it exactly as it writes the real swapchain (the shader
            // does its own sRGB encode into a UNORM image — a _SRGB image here
            // would double-encode). GENERAL layout from creation and forever:
            // TAA imageStores it, the readback copies it, nothing else touches
            // it, so there is no transition to get wrong.
            // COLOR_ATTACHMENT is for the F3 ParticleField billboard composite
            // below: a secondary view runs no overlay pass, so the quads are
            // drawn straight onto this image in a render-pass instance of their
            // own. The image STAYS in GENERAL for that too — GENERAL is a legal
            // colour-attachment layout, so the "one layout forever" property
            // this target was given survives the addition intact.
            v.colorTarget = createStorageImage2D(
                    v.outExt.width, v.outExt.height, ctx->swapchainFormat(),
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    "viewColorTarget");
            // Host-visible landing buffer, allocated with the view rather than
            // per readback, so reading a camera never allocates mid-frame.
            const VkDeviceSize rbBytes =
                    static_cast<VkDeviceSize>(v.outExt.width) * v.outExt.height * 4u;
            v.readbackBuf = createBuffer(
                    ctx->allocator(), ctx->device(), rbBytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);

            // Camera UBO ring + the ReSTIR reservoir pair, both at this view's
            // render extent.
            createCameraUbos();
            createReservoirImages();

            // G-buffer, raster camera UBOs, raster descriptor pool + sets — all
            // sized from renderExtent(), which now answers with v.renderExt.
            ensureHybridResources();

            // This view's own post chain. TaaResolve gets imageCount = 1: its
            // "swapchain" is v.colorTarget, and recordResolve is always called
            // with imageIndex 0. Native-res by scope, so in and out extents are
            // equal and the resolve is a plain 1:1 temporal filter — no
            // upsampling, and no FSR/DLSS (secondaries never take those paths).
            v.taa_ = std::make_unique<vulkan::TaaResolve>(*ctx, cmdPool, 1u, kFramesInFlight);
            v.taa_->createImages(v.renderExt.width, v.renderExt.height,
                                 v.outExt.width, v.outExt.height);
            v.bloom_ = std::make_unique<vulkan::BloomPass>(*ctx, cmdPool, kFramesInFlight);
            v.bloom_->createImages(v.renderExt.width, v.renderExt.height);
            v.post_ = std::make_unique<vulkan::PostComposite>(*ctx, cmdPool, kFramesInFlight);
            if (ctx->rayQuerySupported()) {
                v.deferredShade_ = std::make_unique<vulkan::DeferredShade>(*ctx, kFramesInFlight);
            }

            // Descriptor sets. Per-view objects, so these writes cannot touch
            // anything the primary might have in flight.
            rewriteTaaDescriptors();
            rewriteBloomDescriptors();
            rewriteDeferredDescriptors();

            v.allocatedBytes = vmaAllocatedBytes() - before;
            curView_ = saved;
        }

void VulkanRenderer::Impl::destroySecondaryViewResources(ViewContext& v) {
            VkDevice d = ctx->device();
            // Same rule as a reallocation, one step harder: the images go away
            // entirely, so an armed export has to be released first.
            invalidateFrameInterop(v.id, "removing the view");
            // Passes first, while the device is alive — they own pipelines,
            // pools, samplers and images of their own.
            v.taa_.reset();
            v.post_.reset();
            v.bloom_.reset();
            v.deferredShade_.reset();

            ViewContext* saved = curView_;
            curView_ = &v;
            destroyRasterGbufImages();
            curView_ = saved;

            for (auto& b : v.cameraUbos)         destroyBuffer(ctx->allocator(), b);
            for (auto& b : v.rasterCameraUbos)   destroyBuffer(ctx->allocator(), b);
            for (auto& b : v.drawInfoBuffers)    destroyBuffer(ctx->allocator(), b);
            for (auto& b : v.indirectCmdBuffers) destroyBuffer(ctx->allocator(), b);
            for (auto& i : v.reservoirPosImagesPP) destroyImage2D(ctx->allocator(), d, i);
            for (auto& i : v.reservoirWImagesPP)   destroyImage2D(ctx->allocator(), d, i);
            destroyImage2D(ctx->allocator(), d, v.colorTarget);
            destroyBuffer(ctx->allocator(), v.readbackBuf);
            // One pool destroy frees every set allocated from it.
            if (v.rasterDescPool) vkDestroyDescriptorPool(d, v.rasterDescPool, nullptr);
            v.rasterDescPool = VK_NULL_HANDLE;
            // The splat target names images that were just destroyed; hand the
            // slot back so the next view that asks can have it. Its descriptor
            // sets are per (cloud, target) and are rewritten on the claim.
            if (splat_ && v.splatTarget != vulkan::SplatPass::kNoTarget) {
                splat_->releaseTarget(v.splatTarget);
                v.splatTarget = vulkan::SplatPass::kNoTarget;
            }
        }

void VulkanRenderer::Impl::applyPendingViewChanges() {
            // The shared render pass and pipelines are built during the first
            // render(). A view registered before then simply waits — the flag
            // stays set and this runs again next frame.
            if (rasterGbufRenderPass == VK_NULL_HANDLE) return;

            // Destroys first, so a remove+add in the same gap frees before it
            // allocates rather than peaking at both.
            for (size_t i = views_.size(); i-- > 1;) {
                if (!views_[i]->pendingDestroy) continue;
                // Re-anchor BEFORE destroying: curView_ may still point at the
                // view whose storage is about to go away.
                curView_ = views_[0].get();
                if (!views_[i]->pendingCreate)// never got its resources
                    destroySecondaryViewResources(*views_[i]);
                views_.erase(views_.begin() + static_cast<long>(i));
            }
            for (auto& v : views_) {
                if (!v->pendingCreate) continue;
                v->pendingCreate = false;
                createSecondaryViewResources(*v);
                std::fprintf(stderr, "[threepp] view %u: %ux%u, %.1f MB\n",
                             v->id, v->outExt.width, v->outExt.height,
                             static_cast<double>(v->allocatedBytes) / (1024.0 * 1024.0));
            }
            // A view whose splat flag was set after it was created: claim its
            // SplatPass target here, where the device is drained, rather than
            // from the setter (which is routinely called mid-frame).
            for (auto& v : views_) {
                if (!v->secondary || !v->splats || !v->bloom_) continue;
                if (v->splatTarget != vulkan::SplatPass::kNoTarget) continue;
                curView_ = v.get();
                ensureSplatTarget();
            }
            curView_ = views_[0].get();
            pendingViewChanges_ = false;
        }

uint32_t VulkanRenderer::Impl::addViewImpl(Camera& camera, uint32_t width, uint32_t height) {
            if (width == 0u || height == 0u) return 0u;

            auto v           = std::make_unique<ViewContext>();
            v->secondary     = true;
            v->id            = nextViewId_++;
            v->renderExt     = {width, height};
            v->outExt        = {width, height};
            v->camera        = &camera;
            v->cameraUuid    = camera.uuid;
            v->pendingCreate = true;

            const uint32_t id = v->id;
            views_.push_back(std::move(v));
            pendingViewChanges_ = true;
            return id;
        }

bool VulkanRenderer::Impl::removeViewImpl(uint32_t handle) {
            if (handle == 0u) return false;// 0 is the primary; it is not removable
            for (size_t i = 1; i < views_.size(); ++i) {
                if (views_[i]->id != handle || views_[i]->pendingDestroy) continue;
                views_[i]->pendingDestroy = true;
                pendingViewChanges_       = true;
                return true;
            }
            return false;
        }

bool VulkanRenderer::Impl::setViewCameraImpl(uint32_t handle, Camera& camera) {
            ViewContext* v = findView(handle);
            if (!v || !v->secondary) return false;
            // A different camera is a CUT, not a move: its history describes a
            // viewpoint that no longer exists. Drop it rather than reproject
            // across the discontinuity. Compared by uuid, never by pointer —
            // pointers get recycled here, and a recycled Camera* landing on the
            // same address would inherit the previous camera's history.
            if (v->cameraUuid != camera.uuid) {
                v->prevCameraValid        = false;
                v->rasterPrevVPValid_     = false;
                v->rasterPrevJitterValid_ = false;
                v->deferredCamPrevValid_  = false;
                if (v->taa_) v->taa_->invalidateHistory();
            }
            v->camera     = &camera;
            v->cameraUuid = camera.uuid;
            return true;
        }

// ── Multi-view: per-frame recording ─────────────────────────────────────────

void VulkanRenderer::Impl::recordSecondaryViews(VkCommandBuffer cb) {
            if (views_.size() < 2u) return;

            // Saved so the frame ends exactly where the primary left it. The
            // pane region is primary state (split-screen scissor); a secondary
            // always renders full-frame into its own target.
            const VkExtent2D savedRegionRender = regionRenderExt_;
            const VkExtent2D savedRegionSwap   = regionSwapExt_;
            const int32_t    savedDstX         = regionDstX_;
            const int32_t    savedDstY         = regionDstY_;

            // One query pool per frame-in-flight, one slot pair per pass — a
            // secondary re-running those passes into the same command buffer
            // would overwrite timestamps the primary already wrote. Record
            // silently; lastFrameTimings() stays the primary's.
            gpuTimings_->setSuppressed(true);

            for (size_t i = 1; i < views_.size(); ++i) {
                ViewContext& v = *views_[i];
                // pendingCreate → no resources yet; pendingDestroy → going away
                // at the next frame boundary, so don't spend a frame on it.
                if (v.pendingCreate || v.pendingDestroy) continue;
                if (!v.camera || !v.taa_ || !v.deferredShade_) continue;

                curView_ = &v;
                Camera& cam = *v.camera;

                // Full-frame into this view's own target: no pane offset.
                regionRenderExt_ = v.renderExt;
                regionSwapExt_   = v.outExt;
                regionDstX_      = 0;
                regionDstY_      = 0;

                // Per-camera CPU work. Everything world-space — the TLAS/BLAS
                // build, deformers, lights, materials, motion matrices, the
                // emissive CDF — was already done once for this frame by the
                // primary and is deliberately NOT repeated.
                cam.updateMatrixWorld(true);
                v.orthoFrame_ = cam.is<OrthographicCamera>();
                updateCameraUbo(currentFrame, cam);
                cullEntriesAgainstFrustum(cam);
                uploadRasterCameraUbo(currentFrame, cam);
                buildIndirectDrawData(currentFrame);
                // Sizes/creates this view's G-buffer on the first frame and is
                // a cheap no-op after (the extent never changes for a view).
                ensureHybridResources();

                // G-buffer. imageIndex is meaningless here — a secondary never
                // touches the swapchain — and the debug-blit early-out it feeds
                // is a primary-only path, gated below.
                recordGbufferStage(cb, 0u);

                const VkExtent2D ext   = v.outExt;
                const VkExtent2D ptExt = v.renderExt;
                const float exposure   = currentExposure();
                uint32_t exposureBits;
                std::memcpy(&exposureBits, &exposure, sizeof(exposureBits));
                const float preExp = preExposure();

                recordSceneDispatch(cb, currentFrame, ext, ptExt, exposureBits);

                // Gaussian splats, if this view asked for them: the same place
                // in the frame as the primary's (sceneHdr still linear HDR, the
                // G-buffer depth it tests against final), on this view's own
                // SplatPass target. No-op — not even a barrier — unless
                // setViewSplats(handle, true) was called on this view.
                recordSplats(cb);

                // Built-in TAA tail only — no DLSS, no FSR, no DoF, no overlay,
                // no lens/sensor stage. All of those are primary-only by scope,
                // and recordUpscaleAndPost would branch into them, so the three
                // passes a secondary does need are recorded directly.
                const float effBloomIntensity =
                        bloomIntensity_ / static_cast<float>(std::max(v.bloom_->levels(), 1u));
                v.bloom_->recordPyramid(cb, currentFrame, ptExt.width, ptExt.height,
                                        bloomIntensity_, bloomThreshold_, bloomClamp_);
                v.post_->recordDispatch(cb, currentFrame, ptExt.width, ptExt.height,
                                        static_cast<uint32_t>(toneMapping_),
                                        exposureBits, preExpBits_, envIsBgColor,
                                        effBloomIntensity);
                // imageIndex 0: this view's TaaResolve was built against a
                // one-image "swapchain" that is v.colorTarget.
                v.taa_->recordResolve(cb, currentFrame, /*imageIndex=*/0u,
                                      ptExt.width, ptExt.height,
                                      ext.width, ext.height,
                                      taaBlendAlpha_, 1.0f,
                                      /*sharpen=*/false, 0.f,
                                      v.taaSkyReproj_.data(),
                                      0u, 0u,
                                      ptExt.width, ptExt.height, ext.width, ext.height,
                                      v.taaDepthLin_.data(), /*mblurShutter=*/0.f,
                                      v.taaJitterTexels_[0], v.taaJitterTexels_[1]);

                // ── ParticleField billboards, per view ─────────────────────
                // The one piece of the primary's post-TAA overlay a secondary
                // DOES take, and deliberately: embers and rain are scene
                // content, so a CameraSensor pointed at a campfire has to see
                // them, whereas a wireframe gizmo or a HUD sprite genuinely is
                // a primary-view garnish. The rest of that pass (MSAA inject,
                // resolve, the shared overlayMs* images, overlayInjectSet_)
                // stays primary-only for the reason recordGbufferStage spells
                // out — it is SHARED state and a secondary resizing it
                // corrupts the open command buffer.
                //
                // Same shader, same pipeline layout, 1-sample variant, this
                // view's own camera. Depth comes from this view's G-buffer,
                // which recordGbufferStage left in DEPTH_STENCIL_READ_ONLY —
                // it is the JITTERED depth rather than the primary's unjittered
                // prepass, which is a sub-pixel disagreement on a soft sprite
                // and not worth a second full-screen depth pass per view.
                if (fieldBillboardPipeline1x_ != VK_NULL_HANDLE && sceneHasFieldBillboards()) {
                    // R8/R9: this view's own transmittance prepass, re-
                    // dispatched over the frame's one buffer. T_cam is a
                    // property of the EYE, so the primary's answers are wrong
                    // for a sensor looking from somewhere else — and the
                    // re-dispatch is what makes a shared buffer correct, since
                    // this view's draws sit between this barrier and the next
                    // view's. Outside the rendering scope below, because a
                    // compute dispatch cannot be recorded inside one.
                    recordFieldTransmittance(cb);

                    VkImageMemoryBarrier2 toAtt{};
                    toAtt.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                    toAtt.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    toAtt.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    toAtt.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    toAtt.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    toAtt.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    toAtt.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    toAtt.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toAtt.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    toAtt.image = v.colorTarget.image;
                    toAtt.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    toAtt.subresourceRange.levelCount = 1;
                    toAtt.subresourceRange.layerCount = 1;
                    VkDependencyInfo dIn{};
                    dIn.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dIn.imageMemoryBarrierCount = 1;
                    dIn.pImageMemoryBarriers    = &toAtt;
                    vkCmdPipelineBarrier2(cb, &dIn);

                    VkRenderingAttachmentInfo colorAtt{};
                    colorAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    colorAtt.imageView   = v.colorTarget.view;
                    colorAtt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    colorAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                    colorAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

                    VkRenderingAttachmentInfo depthAtt{};
                    depthAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                    depthAtt.imageView   = v.rasterGbufs[currentFrame].depth.view;
                    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    depthAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
                    depthAtt.storeOp     = VK_ATTACHMENT_STORE_OP_NONE;

                    VkRenderingInfo ri{};
                    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
                    ri.renderArea.offset    = {0, 0};
                    ri.renderArea.extent    = ext;
                    ri.layerCount           = 1;
                    ri.colorAttachmentCount = 1;
                    ri.pColorAttachments    = &colorAtt;
                    ri.pDepthAttachment     = &depthAtt;
                    vkCmdBeginRendering(cb, &ri);
                    VkViewport vp{0.f, 0.f, float(ext.width), float(ext.height), 0.f, 1.f};
                    vkCmdSetViewport(cb, 0, 1, &vp);
                    VkRect2D sc{{0, 0}, ext};
                    vkCmdSetScissor(cb, 0, 1, &sc);
                    // F4: the SHARP quads only — a secondary view gets no
                    // billboard GLOW, and that is a decision rather than an
                    // omission. The glow chain is one half-extent target plus
                    // one pyramid sized to the primary's display, and giving
                    // every view its own would be N targets and N pyramids per
                    // frame for a LENS artefact. A CameraSensor therefore
                    // measures the spark's own radiance (which is the physical
                    // quantity) and not the bloom the display adds around it.
                    // If sensor/display parity on the halo ever matters, the
                    // fix is a per-view BillboardGlowPass instance, not a
                    // change here.
                    recordFieldBillboards(cb, fieldBillboardPipeline1x_, /*glowPass=*/false,
                                          fieldBillboardAlphaPipeline1x_);
                    vkCmdEndRendering(cb);

                    // Back out to everything downstream: recordViewComposite's
                    // copy to the swapchain and readViewRGBPixels' readback both
                    // expect the last write to have been a compute store, and
                    // their own pre-barriers only name that stage.
                    VkImageMemoryBarrier2 fromAtt = toAtt;
                    fromAtt.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
                    fromAtt.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
                    fromAtt.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                            VK_PIPELINE_STAGE_2_COPY_BIT |
                                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                    fromAtt.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                            VK_ACCESS_2_TRANSFER_READ_BIT |
                                            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                    VkDependencyInfo dOut{};
                    dOut.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dOut.imageMemoryBarrierCount = 1;
                    dOut.pImageMemoryBarriers    = &fromAtt;
                    vkCmdPipelineBarrier2(cb, &dOut);
                }
            }

            gpuTimings_->setSuppressed(false);
            regionRenderExt_ = savedRegionRender;
            regionSwapExt_   = savedRegionSwap;
            regionDstX_      = savedDstX;
            regionDstY_      = savedDstY;
            curView_         = views_[0].get();
        }

bool VulkanRenderer::Impl::setViewSensorSurfacesImpl(uint32_t handle, bool enabled) {
            ViewContext* v = findView(handle);
            // The primary never rasterizes sensor-only geometry, so there is no
            // handle 0 case to grant: refuse rather than pretend.
            if (!v || !v->secondary) return false;
            v->sensorSurfaces = enabled;
            return true;
        }

bool VulkanRenderer::Impl::setViewSplatsImpl(uint32_t handle, bool enabled) {
            ViewContext* v = findView(handle);
            // The primary always draws splats, so there is nothing to grant on
            // handle 0: refuse rather than pretend.
            if (!v || !v->secondary) return false;
            v->splats = enabled;
            // Claiming a target slot resizes a shared buffer and writes
            // descriptor sets, so it happens where addView's own resources do —
            // next frame boundary, post-fence, device drained.
            if (enabled && v->splatTarget == vulkan::SplatPass::kNoTarget)
                pendingViewChanges_ = true;
            return true;
        }

bool VulkanRenderer::Impl::setViewDisplayRectImpl(uint32_t handle, int x, int y, int w, int h) {
            ViewContext* v = findView(handle);
            if (!v || !v->secondary) return false;
            v->displayed = w != 0 && h != 0;
            v->dispX = x;
            v->dispY = y;
            v->dispW = w;
            v->dispH = h;
            return true;
        }

// ── Multi-view: putting a view on screen ────────────────────────────────────
//
// A secondary view's image is already on the device, already resolved, and
// already in the swapchain's own format (see createSecondaryViewResources).
// Showing it is therefore one image copy inside the frame's existing command
// buffer — no readback, no upload, no texture, no extra submission. That is the
// whole reason ViewContext::colorTarget was given TRANSFER_SRC usage.
//
// Both images sit in GENERAL for their entire life (the colour target by
// construction; the swapchain from the primary's composite until present), so
// this needs no layout changes at all — only visibility barriers between the
// compute stores that wrote them and the transfer that reads and writes them.
void VulkanRenderer::Impl::recordViewComposite(VkCommandBuffer cb, uint32_t imageIndex) {
            if (views_.size() < 2u) return;

            const VkExtent2D swap = ctx->swapchainExtent();
            if (swap.width == 0u || swap.height == 0u) return;

            struct Job {
                VkImage src;
                VkOffset2D srcOff;
                VkExtent2D srcExt;
                VkOffset2D dstOff;
                VkExtent2D dstExt;
            };
            std::vector<Job> jobs;
            std::vector<VkImageMemoryBarrier2> pre;

            for (size_t i = 1; i < views_.size(); ++i) {
                ViewContext& v = *views_[i];
                if (!v.displayed || v.pendingCreate || v.pendingDestroy) continue;
                if (v.colorTarget.image == VK_NULL_HANDLE) continue;

                const int32_t vw = static_cast<int32_t>(v.outExt.width);
                const int32_t vh = static_cast<int32_t>(v.outExt.height);
                const int32_t dw = v.dispW > 0 ? v.dispW : vw;
                const int32_t dh = v.dispH > 0 ? v.dispH : vh;
                // Only the 1:1 case is composited. Anything else would be a
                // filtered blit of a temporally-resolved image, i.e. a second,
                // different resampling of pixels the view already resolved
                // exactly once — and the caller sizing its view to its rect is
                // both free and what makes the result exact-pixel. A mismatch
                // is a caller bug, so it draws nothing rather than something
                // plausible-looking.
                if (dw != vw || dh != vh) continue;

                // Clip to the swapchain. A dock rect can legitimately hang off
                // the edge for a frame during a window resize, and a copy that
                // runs past the destination is undefined behaviour, not a
                // clipped copy.
                int32_t sx = 0, sy = 0;
                int32_t dx = v.dispX, dy = v.dispY;
                int32_t cw = vw, ch = vh;
                if (dx < 0) { sx = -dx; cw -= sx; dx = 0; }
                if (dy < 0) { sy = -dy; ch -= sy; dy = 0; }
                cw = std::min(cw, static_cast<int32_t>(swap.width) - dx);
                ch = std::min(ch, static_cast<int32_t>(swap.height) - dy);
                if (cw <= 0 || ch <= 0) continue;

                jobs.push_back(Job{v.colorTarget.image,
                                   {sx, sy},
                                   {static_cast<uint32_t>(cw), static_cast<uint32_t>(ch)},
                                   {dx, dy},
                                   {static_cast<uint32_t>(cw), static_cast<uint32_t>(ch)}});

                VkImageMemoryBarrier2 b{};
                b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT;
                b.dstAccessMask       = VK_ACCESS_2_TRANSFER_READ_BIT;
                b.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image               = v.colorTarget.image;
                b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                pre.push_back(b);
            }
            if (jobs.empty()) return;

            const VkImage dst = ctx->swapchainImages()[imageIndex];

            // The swapchain image was last written by the primary's composite
            // (a compute store) or by the scene-capture copy just before this.
            VkImageMemoryBarrier2 dstBarrier{};
            dstBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            dstBarrier.srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                             VK_PIPELINE_STAGE_2_COPY_BIT;
            dstBarrier.srcAccessMask       = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                             VK_ACCESS_2_TRANSFER_READ_BIT;
            dstBarrier.dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT;
            dstBarrier.dstAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            dstBarrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
            dstBarrier.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            dstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            dstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            dstBarrier.image               = dst;
            dstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            dstBarrier.subresourceRange.levelCount = 1;
            dstBarrier.subresourceRange.layerCount = 1;
            pre.push_back(dstBarrier);

            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = static_cast<uint32_t>(pre.size());
            dep.pImageMemoryBarriers    = pre.data();
            vkCmdPipelineBarrier2(cb, &dep);

            for (const auto& j : jobs) {
                VkImageCopy region{};
                region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.srcSubresource.layerCount = 1;
                region.srcOffset                 = {j.srcOff.x, j.srcOff.y, 0};
                region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.dstSubresource.layerCount = 1;
                region.dstOffset                 = {j.dstOff.x, j.dstOff.y, 0};
                region.extent                    = {j.dstExt.width, j.dstExt.height, 1};
                vkCmdCopyImage(cb, j.src, VK_IMAGE_LAYOUT_GENERAL,
                               dst, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
            }

            // Hand the swapchain image back to the overlay (a colour
            // attachment) and to anything that samples or stores into it.
            VkImageMemoryBarrier2 post = dstBarrier;
            post.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            post.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_COPY_BIT;
            post.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                 VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo depPost{};
            depPost.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depPost.imageMemoryBarrierCount = 1;
            depPost.pImageMemoryBarriers    = &post;
            vkCmdPipelineBarrier2(cb, &depPost);
        }

std::vector<unsigned char> VulkanRenderer::Impl::readViewPixelsImpl(uint32_t handle) {
            ViewContext* v = findView(handle);
            if (!v || !v->secondary || v->colorTarget.image == VK_NULL_HANDLE) return {};

            const uint32_t w = v->outExt.width, h = v->outExt.height;
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(w) * h * 4u;

            // The view's target is written by the frame's own submission; wait
            // for it rather than racing. Same trade-off readRGBPixels makes.
            vkDeviceWaitIdle(ctx->device());

            VkCommandBuffer cb = beginOneShot();
            // colorTarget lives in GENERAL permanently (see its creation), so
            // the copy needs no layout change — only a visibility barrier from
            // the TAA store.
            VkImageMemoryBarrier2 b{};
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image         = v->colorTarget.image;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cb, &dep);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent                 = {w, h, 1};
            vkCmdCopyImageToBuffer(cb, v->colorTarget.image, VK_IMAGE_LAYOUT_GENERAL,
                                   v->readbackBuf.handle, 1, &region);
            endAndSubmitOneShot(cb, "readViewRGBPixels");

            // BGRA8_UNORM → tightly packed RGB8. NO vertical flip: the Vulkan
            // readback is already top-down, exactly like readRGBPixels.
            std::vector<unsigned char> rgb(static_cast<size_t>(w) * h * 3u);
            void* mapped = nullptr;
            check(vmaMapMemory(ctx->allocator(), v->readbackBuf.alloc, &mapped),
                  "vmaMapMemory(readViewRGBPixels)");
            invalidateHostReads(ctx->allocator(), v->readbackBuf.alloc, 0, bytes);
            const auto* src = static_cast<const unsigned char*>(mapped);
            const size_t px = static_cast<size_t>(w) * h;
            for (size_t i = 0; i < px; ++i) {
                rgb[i * 3 + 0] = src[i * 4 + 2];
                rgb[i * 3 + 1] = src[i * 4 + 1];
                rgb[i * 3 + 2] = src[i * 4 + 0];
            }
            vmaUnmapMemory(ctx->allocator(), v->readbackBuf.alloc);
            return rgb;
        }

// ── Frame zero-copy interop (Vulkan → CUDA) ─────────────────────────────────
// The "frames out" direction: exported buffers, filled by copies recorded in
// the frame's own command buffer. The caller-facing contract — sync, single
// buffering, invalidation — is documented on VulkanRenderer::enableFrameInterop.

bool VulkanRenderer::Impl::frameInteropSource(uint32_t viewHandle,
                                              VulkanRenderer::FrameChannel channel,
                                              uint32_t gbufSlot, uint32_t imageIndex,
                                              FrameInteropSource& out) {
            out = FrameInteropSource{};
            ViewContext* v = viewHandle == 0u ? &primaryView() : findView(viewHandle);
            if (!v) return false;

            using FC = VulkanRenderer::FrameChannel;
            if (channel == FC::Color) {
                if (v->secondary) {
                    // A secondary's finished frame lands in its own colour
                    // target, which lives in GENERAL from creation and forever
                    // (see createSecondaryViewResources) — no transition to get
                    // wrong, only a visibility barrier, exactly like
                    // readViewPixelsImpl.
                    if (v->colorTarget.image == VK_NULL_HANDLE) return false;
                    out.image      = v->colorTarget.image;
                    out.restLayout = VK_IMAGE_LAYOUT_GENERAL;
                    out.width      = v->outExt.width;
                    out.height     = v->outExt.height;
                    out.bpp        = 4;
                    out.bgra       = v->colorTarget.format == VK_FORMAT_B8G8R8A8_UNORM;
                    return out.width != 0 && out.height != 0;
                }
                // The primary reads the swapchain image the frame is drawing
                // into, at the scene-capture point: post-TAA, pre-overlay.
                // Copying out of the swapchain needs TRANSFER_SRC usage, which
                // the surface may not offer (the same gate recordSceneCapture
                // checks) — an unexportable channel is skipped, not fatal.
                if (!ctx->swapchainSupportsTransferSrc()) return false;
                if (imageIndex >= ctx->swapchainImages().size()) return false;
                const VkExtent2D ext = ctx->swapchainExtent();
                if (ext.width == 0 || ext.height == 0) return false;
                out.image      = ctx->swapchainImages()[imageIndex];
                out.restLayout = VK_IMAGE_LAYOUT_GENERAL;
                out.width      = ext.width;
                out.height     = ext.height;
                out.bpp        = 4;
                out.bgra       = ctx->swapchainFormat() == VK_FORMAT_B8G8R8A8_UNORM;
                return true;
            }

            // The G-buffer AOVs. Attachment, aspect and resting layout are
            // readViewGBufferAOVs' table VERBATIM — depth carries the depth
            // aspect and DEPTH_STENCIL_READ_ONLY, the splat-depth AOV lives in
            // GENERAL for its whole life, every other colour AOV rests in
            // SHADER_READ_ONLY (the raster render pass' finalLayout, which the
            // MSAA resolve also leaves the resolved images in).
            if (v->rasterGbufs.empty()) return false;
            const auto& g = v->rasterGbufs[gbufSlot % v->rasterGbufs.size()];
            const vulkan::Image2D* img = nullptr;
            out.aspect     = VK_IMAGE_ASPECT_COLOR_BIT;
            out.restLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            switch (channel) {
                case FC::Depth:
                    img        = &g.depth;
                    out.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                    out.restLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    break;
                case FC::Normal: img = &g.normal; break;
                case FC::Motion: img = &g.motion; break;
                case FC::Ids:    img = &g.ids;    break;
                case FC::Albedo: img = &g.albedo; break;
                case FC::SplatDepth:
                    // Skipped rather than exported as the 1x1 placeholder, for
                    // the reason readViewGBufferAOVs gives: a caller who forgot
                    // setSplatDepthAov would otherwise get a successful
                    // one-pixel read that looks like "no splats in the frame".
                    // ALLOCATED, not "asked for": the overlay-occlusion latch
                    // allocates it full-res on its own, and those frames were
                    // being skipped as if nothing existed. See the same fix in
                    // readViewGBufferAOVs.
                    if (!splatDepthAovAllocated()) return false;
                    img = &g.splatDepth;
                    out.restLayout = VK_IMAGE_LAYOUT_GENERAL;
                    break;
                default: return false;
            }
            if (!img || img->image == VK_NULL_HANDLE || img->width == 0 || img->height == 0) {
                return false;
            }
            out.image  = img->image;
            out.width  = img->width;
            out.height = img->height;
            // RGBA16 (normal / motion / ids) is 8 bytes; D32_SFLOAT (depth,
            // splat depth) and RGBA8_UNORM (albedo) are 4.
            out.bpp = (img->format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                       img->format == VK_FORMAT_R16G16B16A16_UINT)
                              ? 8u
                              : 4u;
            return true;
        }

std::vector<VulkanRenderer::FrameInteropExport>
VulkanRenderer::Impl::enableFrameInterop(uint32_t viewHandle,
                                         const std::vector<VulkanRenderer::FrameChannel>& channels) {
            std::vector<VulkanRenderer::FrameInteropExport> out;
            if (!ctx || channels.empty()) return out;
            if (!ctx->externalMemorySupported()) {
                std::cerr << "[VulkanRenderer] enableFrameInterop: this device has no "
                             "external-memory extension - use readGBufferAOVs (host readback) "
                             "instead.\n";
                return out;
            }
            // The same "a real frame's contents are in there" predicate the AOV
            // readback uses: the attachments are allocated with the swapchain,
            // so their handles and extents look valid long before anything has
            // been rendered into them.
            if (frameSerial_ == 0 || !sceneBuilt_) return out;
            ViewContext* v = viewHandle == 0u ? &primaryView() : findView(viewHandle);
            if (!v || v->rasterGbufs.empty() || v->rasterGbufs[0].width == 0) return out;

            // Re-enabling replaces the channel set. Tear the old exports down
            // first (device-idle, because an in-flight frame may still be
            // copying into them) so a re-arm never leaks an allocation.
            disableFrameInterop(viewHandle);

            // Size from the FRESHEST G-buffer slot, the same arithmetic
            // readViewGBufferAOVs uses. Every slot is allocated at one extent,
            // so this only matters for the format lookup, but taking the same
            // slot keeps the two paths reading the same table.
            const uint32_t n    = static_cast<uint32_t>(v->rasterGbufs.size());
            const uint32_t slot = (currentFrame + n - 1u) % n;

            FrameInteropView state;
            state.viewHandle = viewHandle;
            for (const auto ch : channels) {
                bool dup = false;
                for (const auto& c : state.channels) dup = dup || c.channel == ch;
                if (dup) continue;// duplicates collapse to one export

                FrameInteropSource src{};
                if (!frameInteropSource(viewHandle, ch, slot, frameImageIndex_, src)) continue;

                FrameInteropChannel c{};
                c.channel = ch;
                c.width   = src.width;
                c.height  = src.height;
                c.bpp     = src.bpp;
                c.bgra    = src.bgra;
                const VkDeviceSize bytes =
                        static_cast<VkDeviceSize>(src.width) * src.height * src.bpp;
                try {
                    // TRANSFER_DST and nothing else: these are copy
                    // destinations no shader ever binds, and
                    // createExternalBuffer's dedicated allocation carries no
                    // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT to ask for anyway.
                    c.buf = vulkan::createExternalBuffer(ctx->physicalDevice(), ctx->device(),
                                                         bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                } catch (const std::exception& e) {
                    std::cerr << "[VulkanRenderer] enableFrameInterop: export failed (" << e.what()
                              << ") - this channel stays on the host readback path.\n";
                    continue;
                }
                state.channels.push_back(std::move(c));
            }
            if (state.channels.empty()) return out;

            out.reserve(state.channels.size());
            VkDeviceSize totalBytes = 0;
            for (auto& c : state.channels) {
                VulkanRenderer::FrameInteropExport e{};
                e.channel       = c.channel;
                e.osHandle      = vulkan::takeOsHandle(ctx->device(), c.buf);
                e.sizeBytes     = static_cast<size_t>(c.buf.size);
                e.width         = c.width;
                e.height        = c.height;
                e.bytesPerPixel = c.bpp;
                e.bgra          = c.bgra;
                out.push_back(e);
                totalBytes += c.buf.size;
            }
            std::fprintf(stderr,
                         "[VulkanRenderer] frame interop armed on view %u: %zu channel(s), "
                         "%.1f MB exported\n",
                         viewHandle, state.channels.size(),
                         static_cast<double>(totalBytes) / (1024.0 * 1024.0));
            frameInterops_.push_back(std::move(state));
            return out;
        }

void VulkanRenderer::Impl::disableFrameInterop(uint32_t viewHandle) {
            if (!ctx) return;
            for (size_t i = 0; i < frameInterops_.size(); ++i) {
                if (frameInterops_[i].viewHandle != viewHandle) continue;
                // The exports are a transfer DESTINATION for whatever frames
                // are still in flight, and destroying them is a free, not a
                // retire — so drain, exactly as disableVertexInterop does for
                // the same reason. A teardown call, never a per-frame one.
                check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (disableFrameInterop)");
                for (auto& c : frameInterops_[i].channels)
                    vulkan::destroyExternalBuffer(ctx->device(), c.buf);
                frameInterops_.erase(frameInterops_.begin() + static_cast<long>(i));
                return;
            }
        }

void VulkanRenderer::Impl::invalidateFrameInterop(uint32_t viewHandle, const char* why) {
            bool armed = false;
            for (const auto& s : frameInterops_) armed = armed || s.viewHandle == viewHandle;
            if (!armed) return;
            std::cerr << "[VulkanRenderer] frame interop disabled on view " << viewHandle << ": "
                      << why << " reallocates the source images, and they must not be freed "
                                "under a live foreign import. Re-enable and re-import.\n";
            disableFrameInterop(viewHandle);
        }

void VulkanRenderer::Impl::destroyFrameInterops() {
            if (frameInterops_.empty() || !ctx) return;
            for (auto& s : frameInterops_)
                for (auto& c : s.channels)
                    vulkan::destroyExternalBuffer(ctx->device(), c.buf);
            frameInterops_.clear();
        }

bool VulkanRenderer::Impl::syncFrameInterop() {
            if (!ctx || frameSerial_ == 0) return false;
            // The fence of the LAST SUBMITTED frame — endFrame advances
            // currentFrame after submitting, so that is the slot before this
            // one, the same arithmetic readViewGBufferAOVs uses to find the
            // freshest G-buffer. One fence, not vkDeviceWaitIdle: a single
            // queue signals fences in submission order, so this retires every
            // earlier frame with it. The frame loop waits this same fence again
            // at its next begin, which is harmless.
            const uint32_t n    = kFramesInFlight;
            const uint32_t slot = (currentFrame + n - 1u) % n;
            check(vkWaitForFences(ctx->device(), 1, &inFlight[slot], VK_TRUE, UINT64_MAX),
                  "vkWaitForFences(syncFrameInterop)");
            return true;
        }

void VulkanRenderer::Impl::recordFrameInterop(VkCommandBuffer cb, uint32_t imageIndex) {
            // The cost gate: a frame with nothing armed records not one extra
            // command, not even a barrier.
            if (frameInterops_.empty()) return;

            struct Copy {
                VkImage image;
                VkImageAspectFlags aspect;
                VkImageLayout restLayout;
                VkBuffer dst;
                uint32_t width, height;
            };
            std::vector<Copy> copies;
            for (const auto& s : frameInterops_) {
                for (const auto& c : s.channels) {
                    FrameInteropSource src{};
                    // Slot `currentFrame`: this copy is recorded into the frame
                    // that is WRITING that slot, after its passes, so it is the
                    // just-written G-buffer rather than the previous one. (The
                    // one-shot readback runs after endFrame advanced the
                    // counter, which is why its arithmetic differs.)
                    if (!frameInteropSource(s.viewHandle, c.channel, currentFrame, imageIndex, src))
                        continue;
                    // Extent/format drift would write the wrong number of bytes
                    // into a foreign mapping. The invalidation rule means this
                    // cannot happen; skipping rather than trusting it is one
                    // comparison per channel per frame.
                    if (src.width != c.width || src.height != c.height || src.bpp != c.bpp) continue;
                    copies.push_back({src.image, src.aspect, src.restLayout, c.buf.handle,
                                      src.width, src.height});
                }
            }
            if (copies.empty()) return;

            // restLayout → TRANSFER_SRC for every source in ONE barrier batch,
            // the copies, then one batch back — the recorded-per-frame form of
            // readViewGBufferAOVs' pattern. ALL_COMMANDS on the source side
            // because the batch mixes attachment writes (the G-buffer),
            // compute stores (the swapchain after TAA/post) and transfer writes
            // (a resolved image), and one conservative barrier at the frame's
            // tail is cheaper to get right than four narrow ones.
            std::vector<VkImageMemoryBarrier> toSrc(copies.size());
            for (size_t i = 0; i < copies.size(); ++i) {
                auto& b               = toSrc[i];
                b                     = VkImageMemoryBarrier{};
                b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout           = copies[i].restLayout;
                b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT |
                                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_TRANSFER_WRITE_BIT;
                b.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image               = copies[i].image;
                b.subresourceRange.aspectMask = copies[i].aspect;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
            }
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr,
                                 static_cast<uint32_t>(toSrc.size()), toSrc.data());

            for (const auto& c : copies) {
                VkBufferImageCopy region{};
                region.bufferOffset                = 0;
                region.bufferRowLength             = 0;// tightly packed, like the readback
                region.bufferImageHeight           = 0;
                region.imageSubresource.aspectMask = c.aspect;
                region.imageSubresource.layerCount = 1;
                region.imageExtent                 = {c.width, c.height, 1};
                vkCmdCopyImageToBuffer(cb, c.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       c.dst, 1, &region);
            }

            // Back to the resting layouts, so the overlay tail and the next
            // frame's consumers find every image where they expect it.
            std::vector<VkImageMemoryBarrier> toRest = toSrc;
            for (size_t i = 0; i < toRest.size(); ++i) {
                toRest[i].oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                toRest[i].newLayout     = copies[i].restLayout;
                toRest[i].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                toRest[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                          VK_ACCESS_SHADER_WRITE_BIT |
                                          VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                          VK_ACCESS_TRANSFER_READ_BIT;
            }
            vkCmdPipelineBarrier(cb,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 0, nullptr, 0, nullptr,
                                 static_cast<uint32_t>(toRest.size()), toRest.data());
        }


void VulkanRenderer::Impl::dumpMemoryStats(const char* tag) const {
            if (!ctx || ctx->allocator() == VK_NULL_HANDLE) return;
            const VkPhysicalDeviceMemoryProperties* mp = nullptr;
            vmaGetMemoryProperties(ctx->allocator(), &mp);
            VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
            vmaGetHeapBudgets(ctx->allocator(), budgets);
            VmaTotalStatistics stats{};
            vmaCalculateStatistics(ctx->allocator(), &stats);
            constexpr double MB = 1024.0 * 1024.0;
            std::fprintf(stderr,
                         "[threepp][vk-mem] %s: reserved %.1f MB in %u blocks, "
                         "live %.1f MB in %u allocations\n",
                         tag,
                         stats.total.statistics.blockBytes / MB,
                         stats.total.statistics.blockCount,
                         stats.total.statistics.allocationBytes / MB,
                         stats.total.statistics.allocationCount);
            const uint32_t heaps = mp ? mp->memoryHeapCount : 0u;
            for (uint32_t i = 0; i < heaps; ++i) {
                const bool devLocal =
                        (mp->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
                std::fprintf(stderr,
                             "[threepp][vk-mem]   heap %u%s: usage %.1f MB / budget %.1f MB\n",
                             i, devLocal ? " (device-local)" : "",
                             budgets[i].usage / MB, budgets[i].budget / MB);
            }
            std::fflush(stderr);
        }
}// namespace threepp
