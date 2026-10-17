#include "threepp/renderers/vulkan/ShadowMapPass.hpp"

#include "threepp/renderers/vulkan/DescriptorShadow.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"// kMaxMaterialTextures

#include "VulkanGpuLayouts.hpp"// GpuShadowMapsUbo, DrawInfoGpu

#include <algorithm>
#include <cstring>
#include <string>

namespace threepp::vulkan {

    ShadowDepthArray createShadowDepthArray(VulkanContext& ctx, uint32_t size, uint32_t layers, bool cube,
                                            bool renderable, const char* name) {
        ShadowDepthArray a;
        a.size   = size;
        a.layers = layers;
        a.cube   = cube;

        VkImageCreateInfo ici{};
        ici.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.flags       = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        ici.imageType   = VK_IMAGE_TYPE_2D;
        ici.format      = VK_FORMAT_D32_SFLOAT;
        ici.extent      = {size, size, 1};
        ici.mipLevels   = 1;
        ici.arrayLayers = layers;
        ici.samples     = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
        // DEPTH_STENCIL_ATTACHMENT either way: the layout the maps are sampled
        // in (DEPTH_STENCIL_READ_ONLY) requires it.
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    (renderable ? 0u : VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        applyConcurrentSharing(ici);
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx.allocator(), &ici, &aci, &a.image, &a.alloc, nullptr), "vmaCreateImage(shadow map)");

        VkImageViewCreateInfo vci{};
        vci.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image                       = a.image;
        vci.viewType                    = cube ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vci.format                      = VK_FORMAT_D32_SFLOAT;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = layers;
        check(createImageView(ctx.device(), &vci, nullptr, &a.view), "createImageView(shadow map)");
        if (renderable) {
            a.layerViews.resize(layers);
            for (uint32_t l = 0; l < layers; ++l) {
                VkImageViewCreateInfo lci            = vci;
                lci.viewType                         = VK_IMAGE_VIEW_TYPE_2D;
                lci.subresourceRange.baseArrayLayer  = l;
                lci.subresourceRange.layerCount      = 1;
                check(createImageView(ctx.device(), &lci, nullptr, &a.layerViews[l]),
                      "createImageView(shadow map layer)");
            }
        }
        if (name) {
            ctx.setObjectName(a.image, name);
            ctx.setObjectName(a.view, name);
        }
        return a;
    }

    void destroyShadowDepthArray(VulkanContext& ctx, ShadowDepthArray& a) {
        for (VkImageView v : a.layerViews) {
            unregisterImageView(v);
            vkDestroyImageView(ctx.device(), v, nullptr);
        }
        if (a.view) {
            unregisterImageView(a.view);
            vkDestroyImageView(ctx.device(), a.view, nullptr);
        }
        if (a.image) vmaDestroyImage(ctx.allocator(), a.image, a.alloc);
        a = {};
    }

    ShadowMapPass::ShadowMapPass(VulkanContext& ctx, uint32_t framesInFlight, VkDescriptorSetLayout rasterLayout)
        : ctx_(ctx), slots_(framesInFlight) {
        // The raster set's shape (VulkanCorePipelines: camera UBO, motion
        // matrices, materials, the bindless albedo maps, DrawInfo); the shadow
        // shaders read 2, 3 and 4.
        VkDescriptorPoolSize sizes[3]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[0].descriptorCount = framesInFlight;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[1].descriptorCount = framesInFlight * 3u;
        sizes[2].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[2].descriptorCount = framesInFlight * kMaxMaterialTextures;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &pool_), "vkCreateDescriptorPool(shadow maps)");
        for (auto& s : slots_) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &rasterLayout;
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, &s.rasterSet), "vkAllocateDescriptorSets(shadow maps)");
            s.ubo = createBuffer(ctx_.allocator(), ctx_.device(), sizeof(impl::GpuShadowMapsUbo),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), s.ubo.alloc, &info);
            s.uboMapped = info.pMappedData;
            std::memset(s.uboMapped, 0, sizeof(impl::GpuShadowMapsUbo));
            flushHostWrites(ctx_.allocator(), s.ubo.alloc);
        }
    }

    ShadowMapPass::~ShadowMapPass() {
        for (auto& s : slots_) {
            freeImages(s);
            destroyBuffer(ctx_.allocator(), s.ubo);
            destroyBuffer(ctx_.allocator(), s.drawInfo);
            destroyBuffer(ctx_.allocator(), s.cmds);
            if (s.rasterSet) forgetDescriptorSet(s.rasterSet);
        }
        if (pool_) vkDestroyDescriptorPool(ctx_.device(), pool_, nullptr);
    }

    void ShadowMapPass::freeImages(Slot& s) {
        destroyShadowDepthArray(ctx_, s.cascades);
        destroyShadowDepthArray(ctx_, s.spots);
        destroyShadowDepthArray(ctx_, s.points);
        s.sizes = {};
    }

    void ShadowMapPass::ensure(uint32_t frame, const Sizes& sizes) {
        Slot& s = slots_[frame];
        if (s.gen != 0 && s.sizes == sizes) return;
        freeImages(s);
        const std::string tag = "shadowMaps[" + std::to_string(frame) + "]";
        if (sizes.cascades > 0)
            s.cascades = createShadowDepthArray(ctx_, sizes.cascadeSize, sizes.cascades, false, true,
                                                (tag + ".cascades").c_str());
        if (sizes.spots > 0)
            s.spots = createShadowDepthArray(ctx_, sizes.spotSize, sizes.spots, false, true, (tag + ".spots").c_str());
        if (sizes.points > 0)
            s.points = createShadowDepthArray(ctx_, sizes.pointSize, 6u * sizes.points, true, true,
                                              (tag + ".points").c_str());
        s.sizes = sizes;
        s.gen   = ++genCounter_;
    }

    bool ShadowMapPass::ensureDrawCapacity(uint32_t frame, uint32_t draws) {
        Slot&              s     = slots_[frame];
        const VkDeviceSize need  = VkDeviceSize(std::max(draws, 1u)) * sizeof(impl::DrawInfoGpu);
        const VkDeviceSize needC = VkDeviceSize(std::max(draws, 1u)) * sizeof(VkDrawIndirectCommand);
        if (s.cmds.size < needC) {
            destroyBuffer(ctx_.allocator(), s.cmds);
            s.cmds = createBuffer(ctx_.allocator(), ctx_.device(), std::max(needC, s.cmds.size * 2u),
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
        if (s.drawInfo.size >= need) return false;
        destroyBuffer(ctx_.allocator(), s.drawInfo);
        s.drawInfo = createBuffer(ctx_.allocator(), ctx_.device(), std::max(need, s.drawInfo.size * 2u),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT);
        VkDescriptorBufferInfo info{};
        info.buffer = s.drawInfo.handle;
        info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = s.rasterSet;
        w.dstBinding      = 4;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo     = &info;
        updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
        return true;
    }

    void ShadowMapPass::writeRasterMaterials(uint32_t frame, VkBuffer materials, const VkDescriptorImageInfo* matTex,
                                             uint32_t matTexCount) {
        Slot&                  s = slots_[frame];
        VkDescriptorBufferInfo mats{};
        mats.buffer = materials;
        mats.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w[2]{};
        w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet          = s.rasterSet;
        w[0].dstBinding      = 2;
        w[0].descriptorCount = 1;
        w[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[0].pBufferInfo     = &mats;
        uint32_t n           = 1;
        if (matTex) {
            w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[1].dstSet          = s.rasterSet;
            w[1].dstBinding      = 3;
            w[1].descriptorCount = matTexCount;
            w[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[1].pImageInfo      = matTex;
            n                    = 2;
            s.matTexWritten      = true;
        }
        updateDescriptorSets(ctx_.device(), n, w, 0, nullptr);
    }

    void ShadowMapPass::record(VkCommandBuffer cb, uint32_t frame, VkPipeline pipeline,
                               VkPipelineLayout layout) const {
        const Slot& s = slots_[frame];
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &s.rasterSet, 0, nullptr);
        vkCmdSetDepthBias(cb, depthBiasConstant, 0.f, depthBiasSlope);
        constexpr VkCullModeFlags kCull[3] = {VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT, VK_CULL_MODE_NONE};
        constexpr VkDeviceSize    kStride  = sizeof(VkDrawIndirectCommand);
        for (const Layer& L : layers) {
            const ShadowDepthArray& arr = L.target == Target::Cascade ? s.cascades
                                        : L.target == Target::Spot    ? s.spots
                                                                      : s.points;
            if (L.layer >= arr.layerViews.size()) continue;
            VkRenderingAttachmentInfo depth{};
            depth.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth.imageView               = arr.layerViews[L.layer];
            depth.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
            depth.clearValue.depthStencil = {0.f, 0u};// reverse-Z: far
            VkRenderingInfo ri{};
            ri.sType             = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.extent = {arr.size, arr.size};
            ri.layerCount        = 1;
            ri.pDepthAttachment  = &depth;
            vkCmdBeginRendering(cb, &ri);
            VkViewport vp{};
            vp.width    = float(arr.size);
            vp.height   = float(arr.size);
            vp.maxDepth = 1.f;
            vkCmdSetViewport(cb, 0, 1, &vp);
            VkRect2D sc{};
            sc.extent = {arr.size, arr.size};
            vkCmdSetScissor(cb, 0, 1, &sc);
            // A point face is drawn without the raster's y flip, so the cube's
            // faces land in the orientation the hardware cube lookup expects;
            // that mirrors the winding too.
            const bool face = L.target == Target::PointFace;
            vkCmdSetFrontFace(cb, face ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE);
            struct {
                float    vp[16];
                uint32_t info[4];
            } pc{};
            std::memcpy(pc.vp, L.vp, sizeof(pc.vp));
            pc.info[0] = face ? 1u : 0u;
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            for (int b = 0; b < 3; ++b) {
                const Group& g = groups[b];
                if (g.count == 0u) continue;
                vkCmdSetCullMode(cb, kCull[b]);
                vkCmdDrawIndirect(cb, s.cmds.handle, VkDeviceSize(g.offset) * kStride, g.count,
                                  static_cast<uint32_t>(kStride));
            }
            vkCmdEndRendering(cb);
        }
    }

}// namespace threepp::vulkan
