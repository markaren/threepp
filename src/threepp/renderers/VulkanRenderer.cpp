// VulkanRenderer — deferred G-buffer renderer (VulkanRenderer::Impl leaf).
//
// Translation-unit layout:
//   vulkan/VulkanCoreImpl.hpp  — CoreImpl struct (shared infra: TLAS/BLAS, scene
//                                fingerprint, raster G-buffer, TAA/bloom, fog, lights,
//                                skinning, ocean/water, LIDAR, env-PMREM, camera UBOs).
//   VulkanRenderer.cpp (this)  — VulkanRenderer::Impl override (DeferredShade +
//                                auto-exposure) + VulkanRendererCore/VulkanRenderer
//                                public method bodies.
//   VulkanPathTracer.cpp       — VulkanPathTracer::Impl override (RT raygen + à-trous
//                                denoise) + VulkanPathTracer public method bodies.
//
// VulkanRenderer (deferred): raster G-buffer supplies primary visibility;
//   deferred_shade.comp lights it analytically and adds ray-query accents
//   (shadows, reflections, AO/GI), denoised + TAA-resolved.
// See VulkanPathTracer.cpp for the iterative path-tracer leaf.

#define VMA_IMPLEMENTATION
#include "vulkan/VulkanCoreImpl.hpp"

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

        // Called once after bloom_->createImages(); wires sceneHdr views.
        void onAfterBloomCreateImages() override {
            if (!autoExposure_) return;
            VkImageView views[kFramesInFlight];
            for (uint32_t f = 0; f < kFramesInFlight; ++f)
                views[f] = bloom_->sceneHdrView(f);
            autoExposure_->rewriteDescriptors(views);
        }

        // CPU per-frame tick: lazy-init, then read histogram + advance EMA.
        void onBeginFrameForPT(uint32_t frame, float dt) override {
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
            autoExposure_->tick(frame, dt);
        }

        // Return adapted exposure when auto-exposure is active.
        [[nodiscard]] float currentExposure() const override {
            if (autoExposureEnabled_ && autoExposure_)
                return autoExposure_->exposure();
            return toneMappingExposure_;
        }

        bool decalsEnabled() const override { return true; }

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
            gpuTimings_->begin(cb, TP_PathTrace, currentFrame);
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
                                           static_cast<float>(glfwGetTime()));
            gpuTimings_->end(cb, TP_PathTrace, currentFrame);// pathTraceMs = deferred SHADE only
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
                deferredShade_->recordDenoiseDispatch(cb, currentFrame, regionRenderExt_.width, regionRenderExt_.height);
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
                                             regionRenderExt_.width, regionRenderExt_.height);
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
        // Only the PT-bound (perspective-camera) primary render() call needs the
        // scene-build pass — it populates lastVisibleEntries_, the BLAS
        // cache, motion bits and the per-mesh fingerprint state the PT
        // pipeline reads. The HUD pattern's second call (ortho camera over
        // a separate HUD scene) must not touch any of that, or it clobbers
        // motionThisFrame_ / meshMovedBits_ / lastVisibleEntries_ and the
        // next PT frame cold-starts (visibly drops to ~1-spp quality).
        // The ortho overlay record path walks the HUD scene directly instead.
        if (!camera.is<OrthographicCamera>() && !secondaryOverlayPane) {
            const auto sceneStart = std::chrono::high_resolution_clock::now();
            core()->ensureSceneBuilt(scene);
            // World-space Sprites (screenSpace == false) are drawn by the overlay
            // billboard pass, not the PT/G-buffer path. Snapshot them each frame
            // with fresh world matrices (ensureSceneBuilt just ran
            // updateMatrixWorld) — independent of the snapshot/lean machinery,
            // since impact sprites move/spawn/expire every frame.
            core()->collectWorldSprites(scene);
            core()->pendingCpuEnsureSceneMs_ =
                    std::chrono::duration<float, std::milli>(
                            std::chrono::high_resolution_clock::now() - sceneStart)
                            .count();
        }
        // Lazy RT pipeline build (first frame). Deferred from constructor so
        // scene-feature spec constants (kSceneFeatures bitmask, e.g. has-glass
        // gating the caustic gather DCE) can be baked into chit with the
        // detected values from ensureSceneBuilt above. Subsequent frames hit
        // the early-out for free.
        if (!core()->rtPipelinesBuilt_) {
            core()->buildAllRtPipelines();
        }
        // After init, also ensure the *current* variant is built. The first-
        // frame build only constructs the active variant; if the user has
        // since toggled into an unbuilt slot, fill it in now (pays a one-time
        // pipeline-compile cost on first toggle, then never again).
        core()->ensureCurrentRtVariantBuilt();
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

    float VulkanRendererCore::getTargetPixelRatio() const { return core()->pixelRatio; }
    void VulkanRendererCore::setPixelRatio(float v) { core()->pixelRatio = v; }

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

    void VulkanRendererCore::clear(bool, bool, bool) {}

    RenderTarget* VulkanRendererCore::getRenderTarget() { return nullptr; }
    void VulkanRendererCore::setRenderTarget(RenderTarget*, int, int) {}

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
        vmaInvalidateAllocation(ctx->allocator(), staging.alloc, 0, bytes);
        vulkan::check(vmaMapMemory(ctx->allocator(), staging.alloc, &mapped),
                      "vmaMapMemory(readRGBPixels)");
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

        vmaInvalidateAllocation(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, 0, bytes);
        void* mapped = nullptr;
        if (vmaMapMemory(impl.ctx->allocator(), impl.sceneCaptureBuf_.alloc, &mapped) != VK_SUCCESS) {
            return {};
        }
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
            // Set up the deterministic shade pipeline — eliminates PT
            // noise as a source of false events. The detector now reads
            // the shade output (eventLumaBuf_) instead of the scene
            // capture buffer, so we don't need to enable sceneCapture
            // for the event camera to work.
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
            // Force the per-pixel motion path to halve FC so the new phase
            // function settles quickly. The fog UBO hash will catch this on
            // the next updateFogUbo call too, but flagging here covers the
            // case where setFogAnisotropy is invoked without changing density.
            core()->motionThisFrame_      = true;
            core()->cameraMovedThisFrame_ = true;
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

    void VulkanRendererCore::setFireflyClamp(float cap) {
        core()->fireflyClamp_ = (cap <= 0.0f) ? 1e30f : cap;
    }

    float VulkanRendererCore::fireflyClamp() const {
        const float v = core()->fireflyClamp_;
        return (v > 1e20f) ? 0.0f : v;
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
            // Toggling changes inscatter every pixel → drop TAA history so the
            // shafts appear/vanish without a ghosted cross-fade.
            pimpl_->motionThisFrame_      = true;
            pimpl_->cameraMovedThisFrame_ = true;
        }
    }

    bool VulkanRenderer::volumetricFog() const {
        return pimpl_->deferredVolFog_;
    }

    void VulkanRenderer::setDeferredStarfield(float intensity) {
        pimpl_->deferredStarIntensity_ = std::max(intensity, 0.f);
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
