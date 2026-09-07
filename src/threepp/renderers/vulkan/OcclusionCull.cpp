#include "threepp/renderers/vulkan/OcclusionCull.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/occl_cull.comp.spv.h"

#include <algorithm>
#include <cstring>

namespace threepp::vulkan {

    namespace {
        struct OcclPc {
            uint32_t drawCount;
            uint32_t mode;
            uint32_t hizMips;
            uint32_t _pad;
            int32_t  hizW, hizH;
        };
        static_assert(sizeof(OcclPc) == 24, "occl_cull push-constant drift");

        constexpr VkDeviceSize kCmdStride   = 4 * sizeof(uint32_t);// VkDrawIndirectCommand
        constexpr VkDeviceSize kVisBytesMin = (1u << 16) / 8;      // 8 KB floor (65 536 instance bits)
    }// namespace

    OcclusionCull::OcclusionCull(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight,
                                 RetireBufferFn retireFn)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight),
          retireFn_(std::move(retireFn)) {
        metaBufs_.resize(framesInFlight_);
        metaPtrs_.resize(framesInFlight_, nullptr);
        cached_.resize(framesInFlight_);
        createPipeline();
        createDummyHiz();

        // visBits: persistent, device-local, one bit per instance
        // (occlCullBitFor domain). Armed ALL-VISIBLE by the first
        // recordFilter (visBitsNeedInit_) so never-tested bits (new
        // objects, first frame, post-growth) draw in phase 1.
        visBits_ = createBuffer(ctx_.allocator(), ctx_.device(), kVisBytesMin,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_MEMORY_USAGE_AUTO);
    }

    OcclusionCull::~OcclusionCull() {
        VkDevice d = ctx_.device();
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (descPool_)   vkDestroyDescriptorPool(d, descPool_, nullptr);
        destroyImage2D(ctx_.allocator(), d, dummyHiz_);
        for (auto& b : metaBufs_) destroyBuffer(ctx_.allocator(), b);
        destroyBuffer(ctx_.allocator(), phase1_);
        destroyBuffer(ctx_.allocator(), phase2_);
        destroyBuffer(ctx_.allocator(), visBits_);
    }

    void OcclusionCull::createDummyHiz() {
        // 1×1 R32F kept in GENERAL forever — bound in the filter set where
        // the pyramid binding is statically declared but never sampled.
        dummyHiz_.width  = 1;
        dummyHiz_.height = 1;
        dummyHiz_.format = VK_FORMAT_R32_SFLOAT;
        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = dummyHiz_.format;
        ici.extent        = {1, 1, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &dummyHiz_.image, &dummyHiz_.alloc, nullptr),
              "vmaCreateImage(occl.dummyHiz)");
        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = dummyHiz_.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = dummyHiz_.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &dummyHiz_.view),
              "vkCreateImageView(occl.dummyHiz)");

        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(occl dummy)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(occl dummy)");
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = dummyHiz_.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        check(vkEndCommandBuffer(cb), "end one-shot cb(occl dummy)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(occl dummy)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(occl dummy)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void OcclusionCull::createPipeline() {
        VkDevice d = ctx_.device();

        VkDescriptorSetLayoutBinding b[6]{};
        auto set = [&](uint32_t i, VkDescriptorType t) {
            b[i].binding = i;
            b[i].descriptorType = t;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        };
        set(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // src indirect records
        set(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // dst indirect records
        set(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // cull meta
        set(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // visBits
        set(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);// farthest HiZ (or dummy)
        set(5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);        // raster camera UBO

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 6;
        dlci.pBindings = b;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(occl)");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = sizeof(OcclPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_),
              "vkCreatePipelineLayout(occl)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kOcclCullCompSpv);
        smci.pCode    = kOcclCullCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(d, &smci, nullptr, &mod), "vkCreateShaderModule(occl_cull)");
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";
        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipeLayout_;
        check(vkCreateComputePipelines(d, ctx_.pipelineCache(), 1, &cpci, nullptr, &pipe_),
              "vkCreateComputePipelines(occl_cull)");
        vkDestroyShaderModule(d, mod, nullptr);

        VkDescriptorPoolSize sizes[3]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        sizes[0].descriptorCount = framesInFlight_ * 2 * 4;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[1].descriptorCount = framesInFlight_ * 2;
        sizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[2].descriptorCount = framesInFlight_ * 2;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = framesInFlight_ * 2;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(occl)");

        std::vector<VkDescriptorSetLayout> layouts(framesInFlight_ * 2, dsLayout_);
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = descPool_;
        dsai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        dsai.pSetLayouts        = layouts.data();
        std::vector<VkDescriptorSet> all(layouts.size());
        check(vkAllocateDescriptorSets(d, &dsai, all.data()), "vkAllocateDescriptorSets(occl)");
        filterSets_.assign(all.begin(), all.begin() + framesInFlight_);
        cullSets_.assign(all.begin() + framesInFlight_, all.end());
    }

    void OcclusionCull::flushMeta(uint32_t frame, uint32_t drawCount) {
        flushHostWrites(ctx_.allocator(), metaBufs_[frame].alloc,
                        0, VkDeviceSize(drawCount) * sizeof(CullMeta));
    }

    void OcclusionCull::retireBuffer(Buffer& b) {
        if (b.handle == VK_NULL_HANDLE) return;
        if (retireFn_) {
            // Deferred: the old handle may still be read by a sibling frame in
            // flight (these buffers are single, not per-fif). The retire queue
            // frees it once its last referencing frame's fence has signaled.
            // retireFn_ zeroes b so the reallocation below can't double-free.
            retireFn_(std::move(b));
        } else {
            // No queue wired (defensive fallback): a full device drain makes the
            // inline free safe. Impl always wires the callback in practice.
            vkDeviceWaitIdle(ctx_.device());
            destroyBuffer(ctx_.allocator(), b);
        }
    }

    void OcclusionCull::ensureCapacity(uint32_t frame, uint32_t drawCount, uint32_t bitDomain) {
        const uint32_t needed = std::max(drawCount, 1u);

        // Per-instance visibility bits — grow with the cull-bit allocator's
        // high-water mark (doubling, so growth stays rare; same shared-
        // buffer reuse rules as phase1/phase2 below). History is lost on
        // growth; the next recordFilter re-arms the new buffer ALL-VISIBLE,
        // which is the conservative reset (one full phase-1 frame).
        const VkDeviceSize visBytes = VkDeviceSize((std::max(bitDomain, 1u) + 31u) / 32u) * 4u;
        if (visBits_.size < visBytes) {
            retireBuffer(visBits_);
            const VkDeviceSize cap = std::max<VkDeviceSize>(visBytes * 2, kVisBytesMin);
            visBits_ = createBuffer(ctx_.allocator(), ctx_.device(), cap,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VMA_MEMORY_USAGE_AUTO);
            visBitsNeedInit_ = true;
        }

        // Per-frame meta buffer (host-mapped, sequential writes).
        const VkDeviceSize metaBytes = VkDeviceSize(needed) * sizeof(CullMeta);
        if (metaBufs_[frame].size < metaBytes) {
            destroyBuffer(ctx_.allocator(), metaBufs_[frame]);
            const VkDeviceSize cap = std::max<VkDeviceSize>(metaBytes * 2, 256 * sizeof(CullMeta));
            metaBufs_[frame] = createBuffer(ctx_.allocator(), ctx_.device(), cap,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VMA_MEMORY_USAGE_AUTO,
                                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                    VMA_ALLOCATION_CREATE_MAPPED_BIT);
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(ctx_.allocator(), metaBufs_[frame].alloc, &info);
            metaPtrs_[frame] = static_cast<CullMeta*>(info.pMappedData);
        }

        // Shared phase buffers (GPU-only; single set — the recording-order
        // barriers serialize cross-frame reuse).
        const VkDeviceSize cmdBytes = VkDeviceSize(needed) * kCmdStride;
        if (capacity_ < needed) {
            retireBuffer(phase1_);
            retireBuffer(phase2_);
            const VkDeviceSize cap = std::max<VkDeviceSize>(cmdBytes * 2, 256 * kCmdStride);
            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            phase1_ = createBuffer(ctx_.allocator(), ctx_.device(), cap, usage,
                                   VMA_MEMORY_USAGE_AUTO);
            phase2_ = createBuffer(ctx_.allocator(), ctx_.device(), cap, usage,
                                   VMA_MEMORY_USAGE_AUTO);
            capacity_ = static_cast<uint32_t>(cap / kCmdStride);
        }
    }

    void OcclusionCull::rewriteSets(uint32_t frame, const FrameInputs& in) {
        auto& c = cached_[frame];
        if (c.srcCmds == in.srcCmds && c.rasterCam == in.rasterCam &&
            c.hizView == in.hizView && c.meta == metaBufs_[frame].handle &&
            c.phase1 == phase1_.handle && c.phase2 == phase2_.handle &&
            c.visBits == visBits_.handle)
            return;
        c = {in.srcCmds, in.rasterCam, in.hizView,
             metaBufs_[frame].handle, phase1_.handle, phase2_.handle,
             visBits_.handle};

        auto writeSet = [&](VkDescriptorSet ds, VkBuffer dst, VkImageView hiz,
                            VkSampler hizSamp, VkImageLayout hizLayout) {
            VkDescriptorBufferInfo bSrc{in.srcCmds, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo bDst{dst, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo bMeta{metaBufs_[frame].handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo bVis{visBits_.handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo bCam{in.rasterCam, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo iHiz{hizSamp, hiz, hizLayout};

            VkWriteDescriptorSet w[6]{};
            auto ww = [&](int n, uint32_t bind, VkDescriptorType t,
                          const VkDescriptorBufferInfo* buf, const VkDescriptorImageInfo* img) {
                w[n].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[n].dstSet          = ds;
                w[n].dstBinding      = bind;
                w[n].descriptorCount = 1;
                w[n].descriptorType  = t;
                w[n].pBufferInfo     = buf;
                w[n].pImageInfo      = img;
            };
            ww(0, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bSrc, nullptr);
            ww(1, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bDst, nullptr);
            ww(2, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bMeta, nullptr);
            ww(3, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bVis, nullptr);
            ww(4, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nullptr, &iHiz);
            ww(5, 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &bCam, nullptr);
            vkUpdateDescriptorSets(ctx_.device(), 6, w, 0, nullptr);
        };
        writeSet(filterSets_[frame], phase1_.handle, dummyHiz_.view, in.hizSampler,
                 VK_IMAGE_LAYOUT_GENERAL);
        writeSet(cullSets_[frame], phase2_.handle, in.hizView, in.hizSampler,
                 VK_IMAGE_LAYOUT_GENERAL);
    }

    void OcclusionCull::prepareFrame(uint32_t frame, uint32_t drawCount, uint32_t bitDomain,
                                     const FrameInputs& in) {
        ensureCapacity(frame, drawCount, bitDomain);
        rewriteSets(frame, in);
    }

    void OcclusionCull::recordFilter(VkCommandBuffer cb, uint32_t frame, uint32_t drawCount) {
        // Arm a fresh/grown visBits buffer ALL-VISIBLE (construction can't —
        // no command buffer there; growth loses history and this is its
        // conservative reset). Rare, so the two dedicated barriers are fine:
        // prior compute use of the old contents → fill, fill → this frame's
        // compute reads.
        if (visBitsNeedInit_) {
            VkMemoryBarrier2 pre{};
            pre.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            pre.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            pre.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            pre.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            pre.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo predep{};
            predep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            predep.memoryBarrierCount = 1;
            predep.pMemoryBarriers    = &pre;
            vkCmdPipelineBarrier2(cb, &predep);

            vkCmdFillBuffer(cb, visBits_.handle, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);

            VkMemoryBarrier2 post{};
            post.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            post.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            post.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            post.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            post.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo postdep{};
            postdep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            postdep.memoryBarrierCount = 1;
            postdep.pMemoryBarriers    = &post;
            vkCmdPipelineBarrier2(cb, &postdep);
            visBitsNeedInit_ = false;
        }

        // Prior frames read phase1/phase2 as indirect commands and this
        // frame's dispatch overwrites phase1 (WAR); also flush the host meta
        // writes to compute reads.
        VkMemoryBarrier2 mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                           VK_PIPELINE_STAGE_2_HOST_BIT;
        mb.srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                           VK_ACCESS_2_HOST_WRITE_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                           VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(cb, &dep);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &filterSets_[frame], 0, nullptr);
        OcclPc pc{drawCount, 0u, 1u, 0u, 1, 1};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (drawCount + 63u) / 64u, 1, 1);

        // Phase-1 writes → pass A's indirect reads.
        VkMemoryBarrier2 tb{};
        tb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        tb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        tb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        tb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        tb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo tdep{};
        tdep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        tdep.memoryBarrierCount = 1;
        tdep.pMemoryBarriers    = &tb;
        vkCmdPipelineBarrier2(cb, &tdep);
    }

    void OcclusionCull::recordCullTest(VkCommandBuffer cb, uint32_t frame, uint32_t drawCount,
                                       uint32_t hizMips, VkExtent2D hizExtent) {
        // The pyramid's trailing barrier already covered its writes → our
        // sampled reads; nothing leading needed (phase2 WAR is covered by
        // the filter's leading barrier earlier this frame... not for
        // phase2 — the PREVIOUS frame's pass-B indirect read of phase2 must
        // finish before this write. The filter's leading barrier this frame
        // already synchronized DRAW_INDIRECT→COMPUTE queue-wide, which
        // includes last frame's pass B, so the WAR is ordered.)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_,
                                0, 1, &cullSets_[frame], 0, nullptr);
        OcclPc pc{drawCount, 1u, hizMips,
                  0u, static_cast<int32_t>(hizExtent.width), static_cast<int32_t>(hizExtent.height)};
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (drawCount + 63u) / 64u, 1, 1);

        // Phase-2 writes → pass B's indirect reads (visBits' next consumer
        // is next frame's filter, whose own leading barrier covers it).
        VkMemoryBarrier2 tb{};
        tb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        tb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        tb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        tb.dstStageMask  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        tb.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo tdep{};
        tdep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        tdep.memoryBarrierCount = 1;
        tdep.pMemoryBarriers    = &tb;
        vkCmdPipelineBarrier2(cb, &tdep);
    }

}// namespace threepp::vulkan
