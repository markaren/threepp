// Shadows: the mode knob (VulkanRenderer::setShadowMode) and the shadow-map
// path's host side — the always-bound stand-ins for bindings 80-83 of the
// deferred set, and the per-view descriptor upkeep.

#include "VulkanCoreImpl.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace threepp {

    VulkanRenderer::ShadowMode VulkanRenderer::Impl::shadowModeFromEnv() {
        const char* e = std::getenv("THREEPP_VK_SHADOW_MODE");
        if (!e) return VulkanRenderer::ShadowMode::Auto;
        const std::string v = e;
        if (v == "rt") return VulkanRenderer::ShadowMode::RayTraced;
        if (v == "sm") return VulkanRenderer::ShadowMode::ShadowMaps;
        if (v == "off") return VulkanRenderer::ShadowMode::Off;
        if (v != "auto") {
            std::fprintf(stderr, "[VulkanRenderer] THREEPP_VK_SHADOW_MODE=%s: expected auto, rt, sm or off; "
                                 "using auto\n", e);
        }
        return VulkanRenderer::ShadowMode::Auto;
    }

    uint32_t VulkanRenderer::Impl::shadowMapSizeCapFromEnv() {
        const char* e = std::getenv("THREEPP_VK_SHADOW_MAPSIZE");
        return e ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }

    VulkanRenderer::ShadowMode VulkanRenderer::Impl::activeShadowMode() const {
        using M = VulkanRenderer::ShadowMode;
        switch (shadowModeReq_) {
            case M::Off: return M::Off;
            case M::ShadowMaps: return M::ShadowMaps;
            default: return ctx->rayQuerySupported() ? M::RayTraced : M::ShadowMaps;
        }
    }

    void VulkanRenderer::Impl::ensureShadowDummies() {
        if (shadowDummyUbo_.handle != VK_NULL_HANDLE) return;
        shadowDummyArray_ = vulkan::createShadowDepthArray(*ctx, 1, 1, false, false, "shadowDummy.array");
        shadowDummyCube_  = vulkan::createShadowDepthArray(*ctx, 1, 6, true, false, "shadowDummy.cube");
        shadowDummyUbo_   = createBuffer(ctx->allocator(), ctx->device(), sizeof(GpuShadowMapsUbo),
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        const GpuShadowMapsUbo zero{};
        uploadHostVisible(ctx->allocator(), shadowDummyUbo_, &zero, sizeof(zero));
        // Cleared to 0 (the far plane: nothing cast) and left where the
        // deferred set's descriptors say they rest.
        VkCommandBuffer cb = beginOneShot();
        for (VkImage img : {shadowDummyArray_.image, shadowDummyCube_.image}) {
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            range.levelCount = 1;
            range.layerCount = VK_REMAINING_ARRAY_LAYERS;
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = img;
            b.subresourceRange    = range;
            b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
            const VkClearDepthStencilValue clear{0.f, 0u};
            vkCmdClearDepthStencilImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
            b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 0, nullptr, 1, &b);
        }
        endAndSubmitOneShot(cb, "shadow map dummies");
    }

    void VulkanRenderer::Impl::destroyShadowResources() {
        vulkan::destroyShadowDepthArray(*ctx, shadowDummyArray_);
        vulkan::destroyShadowDepthArray(*ctx, shadowDummyCube_);
        destroyBuffer(ctx->allocator(), shadowDummyUbo_);
    }

    VulkanRenderer::Impl::ShadowBinding VulkanRenderer::Impl::shadowBindingFor(uint32_t frame) const {
        ShadowBinding b;
        b.cascades = shadowDummyArray_.view;
        b.spots    = shadowDummyArray_.view;
        b.points   = shadowDummyCube_.view;
        b.ubo      = shadowDummyUbo_.handle;
        const auto& v = *curView_;
        if (!v.shadowLive || !v.shadowMaps_) return b;
        const auto& s = v.shadowMaps_->slot(frame);
        if (s.gen == 0) return b;
        if (s.cascades.view) b.cascades = s.cascades.view;
        if (s.spots.view) b.spots = s.spots.view;
        if (s.points.view) b.points = s.points.view;
        b.ubo = s.ubo.handle;
        b.gen = s.gen;
        return b;
    }

    void VulkanRenderer::Impl::syncShadowDescriptors(uint32_t frame) {
        if (!view().deferredShade_) return;
        ensureShadowDummies();
        const ShadowBinding b = shadowBindingFor(frame);
        if (view().shadowBoundGen[frame] == b.gen) return;
        view().deferredShade_->rewriteShadowMaps(frame, b.cascades, b.spots, b.points, b.ubo);
        view().shadowBoundGen[frame] = b.gen;
    }

    void VulkanRenderer::Impl::prepareShadowMaps(Camera& camera) {
        (void) camera;
        view().shadowLive = false;
        syncShadowDescriptors(currentFrame);
    }

}// namespace threepp
