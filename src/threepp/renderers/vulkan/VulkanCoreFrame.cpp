#include "VulkanCoreImpl.hpp"

namespace threepp {

void VulkanRendererCore::CoreImpl::createCommandResources() {
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

void VulkanRendererCore::CoreImpl::createRenderFinishedSemaphores() {
            for (auto s : renderFinished)
                if (s) vkDestroySemaphore(ctx->device(), s, nullptr);
            renderFinished.assign(ctx->swapchainImages().size(), VK_NULL_HANDLE);
            VkSemaphoreCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (auto& s : renderFinished)
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &s), "vkCreateSemaphore B");
        }

VkCommandBuffer VulkanRendererCore::CoreImpl::beginOneShot() {
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
            return cb;
        }

void VulkanRendererCore::CoreImpl::endAndSubmitOneShot(VkCommandBuffer cb, const char* label) {
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

void VulkanRendererCore::CoreImpl::createReservoirImages() {
            // Render extent, not swapchain extent — the per-pixel ReSTIR DI
            // reservoir images are sized to the resolution the deferred shade
            // launches at. Equal to the swapchain extent unless renderScale_ < 1.
            const VkExtent2D ext = renderExtent();
            for (size_t i = 0; i < reservoirPosImagesPP.size(); ++i) {
                reservoirPosImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R32G32B32A32_SFLOAT,
                        0, i == 0 ? "reservoirPosImagePP[0]" : "reservoirPosImagePP[1]");
            }
            for (size_t i = 0; i < reservoirWImagesPP.size(); ++i) {
                reservoirWImagesPP[i] = createStorageImage2D(
                        ext.width, ext.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                        0, i == 0 ? "reservoirWImagePP[0]" : "reservoirWImagePP[1]");
            }
            // Storage memory contents are undefined after vmaCreateImage — clear
            // both reservoir slots to 0 so the first frame's temporal-reuse read
            // sees M=0 (no prior history) instead of garbage.
            clearGbufImages();
            sampleIndex = 0;
            prevCameraValid = false;
            prevWorldMats.clear();
        }

void VulkanRendererCore::CoreImpl::resetAccumulation() {
            sampleIndex = 0;
            prevCameraValid = false;
            prevWorldMats.clear();
            if (frameState_ != FrameState::Idle) {
                pendingAccumulationReset_ = true;
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            clearGbufImages();
        }

void VulkanRendererCore::CoreImpl::scanLidar(const std::vector<LidarBeam>& beams,
                       std::vector<LidarReturn>& outResults,
                       const LidarParams& params) {
            outResults.clear();
            if (beams.empty()) return;

            // Lazy-construct the LIDAR pipeline. Idempotent; if the user
            // never calls scanLidar, the pipeline + SBT cost is never paid.
            if (!lidar_) lidar_ = std::make_unique<vulkan::LidarScanner>(*ctx);

            // Drain all in-flight frames so the TLAS + geom/mat buffers
            // are stable while we read them.
            vkDeviceWaitIdle(ctx->device());

            // Force-flush MaterialDescs into buffer slot 0 — gives us a
            // known-good frame target without having to chase `currentFrame`
            // semantics across render() calls.
            matDescsDirty_[0] = true;
            flushMaterialDescsIfDirty(0);

            // Push constants encode the LIDAR equation parameters. The
            // shader multiplies the raw `laserPower · f_back · cos θ · η / r²`
            // contribution by `invReferenceIntensity` so that, AT laserPower = 1,
            // a perpendicular 1.0-albedo *Lambertian* surface at `referenceRange`
            // reads as 1.0. The reference is purely geometric (π · refRange²)
            // — it does NOT include laserPower, so raising the power slider
            // scales every return linearly (the whole point of the knob).
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

            // results[(beam * samplesPerBeam + sample) * maxReturns + slot]
            const size_t totalSlots = beams.size() * pc.samplesPerBeam * pc.maxReturns;
            std::vector<vulkan_lidar::LidarResult> raw(totalSlots);

            // Pick the most recently uploaded fog slot. render() bumps
            // currentFrame at the end, so the just-used slot is
            // (currentFrame - 1) mod N. The slot's contents reflect the
            // scene's current fog state because updateFogUbo runs every
            // frame from render().
            const uint32_t fogSlot = (currentFrame + kFramesInFlight - 1) % kFramesInFlight;

            lidar_->scan(ctx->graphicsQueue(),
                         tlas,
                         geometryDescsBuffer.handle, geometryDescsBuffer.size,
                         materialDescsBuffers[0].handle, materialDescsBuffers[0].size,
                         fogUbos[fogSlot].handle, fogUbos[fogSlot].size,
                         pc,
                         packed.data(), static_cast<uint32_t>(packed.size()),
                         raw.data());

            // Unpack into the public LidarReturn struct. We preserve the
            // fixed-stride layout (numBeams * maxReturns) — caller filters
            // entries with hitInstanceId < 0.
            outResults.resize(totalSlots);
            for (size_t i = 0; i < totalSlots; ++i) {
                const auto& r = raw[i];
                auto& o = outResults[i];
                o.position.set(r.position[0], r.position[1], r.position[2]);
                o.normal.set(r.normal[0], r.normal[1], r.normal[2]);
                o.distance      = r.distance;
                o.intensity     = r.intensity;
                o.hitInstanceId = r.instanceId;
                o.returnNo      = r.returnNo;
            }
        }

void VulkanRendererCore::CoreImpl::reallocateRenderExtentResources() {
            for (auto& img : reservoirPosImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
            for (auto& img : reservoirWImagesPP) destroyImage2D(ctx->allocator(), ctx->device(), img);
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
                const VkExtent2D outExt = ctx->swapchainExtent();
                taa_->createImages(inExt.width, inExt.height,
                                   outExt.width, outExt.height);
            }
            bloom_->createImages(renderExtent().width, renderExtent().height);
            onAfterBloomCreateImages();
            // TAA descriptor sets are persistent (pool lives inside TaaResolve);
            // just rewrite them to the new image / view handles.
            rewriteTaaDescriptors();
            rewriteBloomDescriptors();// gbuf + TAA-input views changed (also
                                      // covers the HDR-mode lazy alloc — see
                                      // its own taaHdrInput_ gate)
            taaHdrPlumbingDirty_ = false;// handled by the call just above
            rewriteDeferredDescriptors();// raster gbuf + sceneHdr views changed
        }

void VulkanRendererCore::CoreImpl::recreateSwapchainAndDescriptors() {
            ctx->recreateSwapchain();// device-idles internally
            createRenderFinishedSemaphores();
            reallocateRenderExtentResources();
            size = WindowSize{static_cast<int>(ctx->swapchainExtent().width),
                              static_cast<int>(ctx->swapchainExtent().height)};
        }

void VulkanRendererCore::CoreImpl::setRenderScale(float scale) {
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

void VulkanRendererCore::CoreImpl::setGbufferMsaa(uint32_t samples) {
            const uint32_t clamped = samples >= 4 ? 4u : (samples >= 2 ? 2u : 1u);
            if (clamped == gbufMsaaSamples_) return;
            gbufMsaaSamples_ = clamped;
            if (frameState_ != FrameState::Idle) {
                pendingRenderScaleRealloc_ = true;// shares the reallocation gate
                return;
            }
            vkDeviceWaitIdle(ctx->device());
            reallocateRenderExtentResources();
        }

void VulkanRendererCore::CoreImpl::beginCommandRecording(VkCommandBuffer cb) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            gpuTimings_->beginFrame(cb, currentFrame);
        }

bool VulkanRendererCore::CoreImpl::beginDeferredFrame(Object3D& scene, Camera& camera) {
            VkDevice d = ctx->device();
            vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);
            // Fence has signaled → the previous render that wrote into this
            // frame's query pool has retired. Read it now, before we reset
            // the pool and re-record. Result is stored in gpuTimings_ for
            // the public getter to read.
            gpuTimings_->readBack(currentFrame, pendingCpuEnsureSceneMs_);

            // Apply any setter requests deferred from mid-frame. setRenderScale
            // / resetAccumulation issue vkDeviceWaitIdle + reallocate descriptor
            // pool / clear images — both unsafe with an open cmd buffer in the
            // prior frame. We're past the fence wait now so the GPU is idle for
            // *this* slot at minimum; vkDeviceWaitIdle below drains the rest.
            if (pendingRenderScaleRealloc_ || pendingAccumulationReset_) {
                vkDeviceWaitIdle(d);
                if (pendingRenderScaleRealloc_) {
                    reallocateRenderExtentResources();// also rewrites deferred descriptors
                    pendingRenderScaleRealloc_ = false;
                }
                if (pendingAccumulationReset_) {
                    clearGbufImages();
                    pendingAccumulationReset_ = false;
                }
            }

            uint32_t imageIndex = 0;
            VkResult acq = vkAcquireNextImageKHR(d, ctx->swapchain(), UINT64_MAX,
                                                 imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return false;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }
            frameImageIndex_ = imageIndex;

            updateCameraUbo(currentFrame, camera);
            updateLightsUbo(currentFrame, scene);
            updateFogUbo(currentFrame, scene, camera);
            // Safe to write motionMatBuffers[currentFrame] now that the
            // inFlight[currentFrame] fence has been signaled — the GPU has
            // finished its previous use of this slot.
            computeAndUploadMotionMatrices(currentFrame, lastVisibleEntries_);
            // Same fence guarantee covers materialDescsBuffers[currentFrame].
            // ensureSceneBuilt staged any material-value change in
            // matDescsCached_ + flipped matDescsDirty_[*]=true; flush this
            // slot now (the other slot flushes when its frame comes around).
            flushMaterialDescsIfDirty(currentFrame);
            // Per-frame frustum cull: tags every entry with `inFrustum`
            // for the raster passes to consume.
            cullEntriesAgainstFrustum(camera);
            // Hybrid raster prepass: lazy-create resources on first use,
            // refresh attachments on resize, then upload the per-frame
            // camera VPs (curr jittered, curr unjittered, prev unjittered).
            // Must run after computeAndUploadMotionMatrices so the descriptor
            // rewrite picks up the populated motionMat buffer for this frame.
            ensureHybridResources();
            uploadRasterCameraUbo(currentFrame, camera);
            // Build the per-frame DrawInfo + indirect-cmd buffers used
            // by the indirect-drawing gbuf pass. Runs after the cull
            // pass + camera upload (depends on both) and before record.
            buildIndirectDrawData(currentFrame);
            uploadMeshMovedBits(currentFrame);
            // Same fence guarantees emissiveTriBuffers[currentFrame] is no
            // longer in use; rebuild the per-frame CDF and rewrite the emissive
            // binding if the buffer grew.
            if (buildAndUploadEmissiveTris(currentFrame, lastVisibleEntries_)) {
                // Keep the deferred pass's emissive binding fresh when the
                // per-frame buffer grows.
                if (deferredShade_) {
                    deferredShade_->rewriteEmissive(currentFrame,
                                                    emissiveTriBuffers[currentFrame].handle);
                    probeGI_->rewriteEmissive(currentFrame,
                                              emissiveTriBuffers[currentFrame].handle);
                }
            }
            if (refreshEnvTextureFromScene(scene)) {
                rewriteDeferredDescriptors();
                // Env is a primary radiance source — can't reproject, so wipe
                // the ReSTIR DI reservoir history so the next frame cold-starts
                // instead of reusing samples lit by the old environment.
                vkDeviceWaitIdle(ctx->device());
                clearGbufImages();
            }

            vkResetFences(d, 1, &inFlight[currentFrame]);
            // Per-frame hook: CPU readbacks (e.g. auto-exposure histogram) that are
            // safe only after the current slot's prior GPU work is retired (fence above).
            {
                const double now = glfwGetTime();
                const float  dt  = (lastFrameTime_ > 0.0)
                                   ? static_cast<float>(now - lastFrameTime_) : 0.016f;
                lastFrameTime_ = now;
                onBeginDeferredFrame(currentFrame, dt);
            }
            vkResetCommandBuffer(cmdBuffers[currentFrame], 0);
            beginCommandRecording(cmdBuffers[currentFrame]);
            // Record the full deferred-render body into the now-open cmd
            // buffer. Leaves the swapchain image in GENERAL.
            recordCommandBuffer(cmdBuffers[currentFrame], imageIndex);

            // Scene-only swapchain capture. Runs BEFORE the sprite + ImGui
            // overlays composite — sensor pipelines that consume the
            // renderer's output need a clean image without their own
            // visualisation drawn on top of it, otherwise they'd see
            // their own readout as scene motion and feedback-loop.
            if (sceneCaptureEnabled_) {
                recordSceneCapture(cmdBuffers[currentFrame], imageIndex);
            }

            // GPU event camera detection. Run event_shade first to fill
            // eventLumaBuf_ with deterministic Lambert lighting from the
            // gbuf (eliminates stochastic shading noise as a source of false
            // events), then dispatch the detector against that buffer.
            if (eventCamEnabled_ && eventCam_ &&
                eventShadePipeline_ != VK_NULL_HANDLE) {
                recordEventShade(cmdBuffers[currentFrame], currentFrame);
                eventCam_->record(cmdBuffers[currentFrame],
                                  eventLumaBuf_.handle,
                                  eventCamParams_);
            }

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

bool VulkanRendererCore::CoreImpl::beginFrameOrthoOnly() {
            VkDevice d = ctx->device();
            vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);
            gpuTimings_->readBack(currentFrame, pendingCpuEnsureSceneMs_);

            uint32_t imageIndex = 0;
            VkResult acq = vkAcquireNextImageKHR(d, ctx->swapchain(), UINT64_MAX,
                                                 imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
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

void VulkanRendererCore::CoreImpl::endFrame() {
            if (frameState_ == FrameState::Idle) return;

            const uint32_t imageIndex = frameImageIndex_;
            VkCommandBuffer cb = cmdBuffers[currentFrame];

            recordOverlayAndPresentTransition(cb, imageIndex);
            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            gpuTimings_->finishRecord();

            VkSemaphoreSubmitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfo.semaphore = imageAvailable[currentFrame];
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

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

            VkSubmitInfo2 submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submit.waitSemaphoreInfoCount = 1;
            submit.pWaitSemaphoreInfos = &waitInfo;
            submit.commandBufferInfoCount = 1;
            submit.pCommandBufferInfos = &cbInfo;
            submit.signalSemaphoreInfoCount = 1;
            submit.pSignalSemaphoreInfos = &signalInfo;
            check(vkQueueSubmit2(ctx->graphicsQueue(), 1, &submit, inFlight[currentFrame]),
                  "vkQueueSubmit2");

            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &renderFinished[imageIndex];
            VkSwapchainKHR sc = ctx->swapchain();
            pi.swapchainCount = 1;
            pi.pSwapchains = &sc;
            pi.pImageIndices = &imageIndex;

            VkResult pr = vkQueuePresentKHR(ctx->presentQueue(), &pi);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || needsResize) {
                needsResize = false;
                recreateSwapchainAndDescriptors();
            } else if (pr != VK_SUCCESS) {
                check(pr, "vkQueuePresentKHR");
            }

            // Cap to keep `subIdx = sampleIndex * spp + s` (deferred_shade.comp)
            // from overflowing uint32. With spp ≤ 256, cap at 2^24 leaves headroom
            // (16M·256 ≈ 4G, just under uint32 max). The previous 65535 cap froze
            // the blue-noise jitter and Halton sequence after ~18 min at 60 fps,
            // causing the per-pixel FC=4096 running mean to slowly absorb the
            // single deterministic sample — image visibly resets from converged
            // toward biased noise. 16M ≈ 75 hours at 60 fps, no longer reachable.
            if (sampleIndex < (1u << 24)) ++sampleIndex;

            currentFrame  = (currentFrame + 1) % kFramesInFlight;
            frameState_   = FrameState::Idle;
        }

void VulkanRendererCore::CoreImpl::renderFrame(Object3D& scene, Camera& camera) {
            const bool isOrtho = camera.is<OrthographicCamera>();

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
                    if (!beginDeferredFrame(scene, camera)) return;
                    frameState_ = FrameState::RecordingPostShade;
                }
                return;
            }

            // Frame already in flight from a prior render() call this iteration.
            if (!isOrtho) {
                // Split-screen secondary pane: when a scissor sub-rect is set, a
                // second perspective render() composes overlay-only (Points /
                // Lines / Sprites of THIS scene+camera) into that region of the
                // open frame, beside the primary deferred-render pane — without
                // touching the accumulation/TLAS state of that pane. Without a
                // scissor, fall back to the old behavior (finalize the prior
                // frame, restart for this one).
                if (scissorTest && scissor.z >= 1.f && scissor.w >= 1.f) {
                    const VkExtent2D full = ctx->swapchainExtent();
                    const uint32_t rx = static_cast<uint32_t>(std::clamp(static_cast<int>(scissor.x), 0, static_cast<int>(full.width)));
                    const uint32_t rw = static_cast<uint32_t>(std::clamp(static_cast<int>(scissor.z), 1, static_cast<int>(full.width) - static_cast<int>(rx)));
                    const uint32_t rh = static_cast<uint32_t>(std::clamp(static_cast<int>(scissor.w), 1, static_cast<int>(full.height)));
                    const int      syB = std::clamp(static_cast<int>(scissor.y), 0, static_cast<int>(full.height) - static_cast<int>(rh));
                    const uint32_t ry = static_cast<uint32_t>(static_cast<int>(full.height) - (syB + static_cast<int>(rh)));
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

}// namespace threepp
