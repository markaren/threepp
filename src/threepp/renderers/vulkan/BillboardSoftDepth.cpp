#include "threepp/renderers/vulkan/BillboardSoftDepth.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/particlefield_soft_depth.comp.ms.spv.h"
#include "threepp/renderers/vulkan/shaders/particlefield_soft_depth.comp.spv.h"

#include <algorithm>
#include <cstring>

namespace threepp::vulkan {

    namespace {

        // Mirrors Pc in particlefield_soft_depth.comp.
        struct SoftDepthPc {
            VkDeviceAddress outAddr;
            std::uint32_t   width;
            std::uint32_t   height;
            float           lin[4];
        };
        static_assert(sizeof(SoftDepthPc) == 32, "SoftDepthPc drifted from particlefield_soft_depth.comp");

        constexpr std::uint32_t kLocalSize = 8;// local_size_x/y of the shader
        // Sets the pool can hold at once: this many views, each with one set per
        // frame in flight. A view past it publishes no target and keeps the hard
        // edge, which is what a missing feature looks like here, not a crash.
        constexpr std::uint32_t kMaxViews = 16;

        VkPipeline makePipe(VkDevice d, VkPipelineCache cache, VkPipelineLayout layout,
                            const std::uint32_t* spv, std::size_t bytes, const char* label) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes;
            smci.pCode    = spv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(d, &smci, nullptr, &mod), label);

            VkComputePipelineCreateInfo cpci{};
            cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = mod;
            cpci.stage.pName  = "main";
            cpci.layout       = layout;
            VkPipeline pipe = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(d, cache, 1, &cpci, nullptr, &pipe), label);
            vkDestroyShaderModule(d, mod, nullptr);
            return pipe;
        }

    }// namespace

    BillboardSoftDepth::BillboardSoftDepth(VulkanContext& ctx, std::uint32_t framesInFlight,
                                           RetireBufferFn retireFn)
        : ctx_(ctx), framesInFlight_(framesInFlight), retireFn_(std::move(retireFn)) {
        const VkDevice d = ctx_.device();

        // texelFetch ignores filtering; a combined image sampler still needs a
        // sampler object, and NEAREST states the intent.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_NEAREST;
        sci.minFilter    = VK_FILTER_NEAREST;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 0.f;
        check(vkCreateSampler(d, &sci, nullptr, &sampler_), "vkCreateSampler(bbsoft)");

        VkDescriptorSetLayoutBinding bnd{};
        bnd.binding         = 0;
        bnd.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bnd.descriptorCount = 1;
        bnd.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 1;
        dlci.pBindings    = &bnd;
        check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(bbsoft)");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size       = sizeof(SoftDepthPc);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        check(vkCreatePipelineLayout(d, &plci, nullptr, &pipeLayout_), "vkCreatePipelineLayout(bbsoft)");

        // Both variants up front: which one runs depends on whether the source
        // depth is multisampled, which differs between the primary (the Canvas
        // antialiasing) and a secondary view (always one sample).
        pipe_   = makePipe(d, ctx_.pipelineCache(), pipeLayout_, kParticleFieldSoftDepthCompSpv,
                           sizeof(kParticleFieldSoftDepthCompSpv), "vkCreateComputePipelines(bbsoft)");
        pipeMs_ = makePipe(d, ctx_.pipelineCache(), pipeLayout_, kParticleFieldSoftDepthMsCompSpv,
                           sizeof(kParticleFieldSoftDepthMsCompSpv), "vkCreateComputePipelines(bbsoft.ms)");

        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = kMaxViews * framesInFlight_;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        // FREE: a view that goes away gives its sets back (prune()).
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = kMaxViews * framesInFlight_;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes    = &ps;
        check(vkCreateDescriptorPool(d, &dpci, nullptr, &pool_), "vkCreateDescriptorPool(bbsoft)");
    }

    // The renderer destroys this with the device idle, after its own retire
    // queue has drained, so everything goes straight back.
    BillboardSoftDepth::~BillboardSoftDepth() {
        const VkDevice d = ctx_.device();
        for (auto& v : views_)
            for (auto& s : v.slots) destroyBuffer(ctx_.allocator(), s.buf);
        if (pool_)       vkDestroyDescriptorPool(d, pool_, nullptr);// frees every set
        if (pipe_)       vkDestroyPipeline(d, pipe_, nullptr);
        if (pipeMs_)     vkDestroyPipeline(d, pipeMs_, nullptr);
        if (pipeLayout_) vkDestroyPipelineLayout(d, pipeLayout_, nullptr);
        if (dsLayout_)   vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (sampler_)    vkDestroySampler(d, sampler_, nullptr);
    }

    void BillboardSoftDepth::release(ViewSlots& v) {
        for (auto& s : v.slots) {
            if (s.buf.handle != VK_NULL_HANDLE) {
                if (retireFn_) retireFn_(std::move(s.buf));
                else destroyBuffer(ctx_.allocator(), s.buf);
                s.buf = Buffer{};
            }
            if (s.set != VK_NULL_HANDLE) vkFreeDescriptorSets(ctx_.device(), pool_, 1, &s.set);
            s.set = VK_NULL_HANDLE;
        }
    }

    // A view entry nothing has prepared for a few frames: the view was removed,
    // or the scene's last soft field went away. More frames than are in flight
    // have passed since its last use, so no command buffer still names its sets
    // and freeing them is safe; the buffers go through the retire queue anyway.
    void BillboardSoftDepth::prune(std::uint64_t serial) {
        const std::uint64_t keep = std::uint64_t(framesInFlight_) + 2u;
        for (auto it = views_.begin(); it != views_.end();) {
            if (it->lastSerial + keep < serial) {
                release(*it);
                it = views_.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool BillboardSoftDepth::prepare(std::uint32_t viewId, std::uint32_t frame, std::uint64_t serial,
                                     VkExtent2D ext, VkImageView depthView, bool multisampled,
                                     const float projInverse[16], Dispatch& out) {
        out = Dispatch{};
        if (frame >= framesInFlight_ || depthView == VK_NULL_HANDLE || ext.width == 0 || ext.height == 0)
            return false;
        prune(serial);

        auto it = std::find_if(views_.begin(), views_.end(),
                               [&](const ViewSlots& v) { return v.viewId == viewId; });
        if (it == views_.end()) {
            ViewSlots v;
            v.viewId = viewId;
            v.slots.resize(framesInFlight_);
            views_.push_back(std::move(v));
            it = views_.end() - 1;
        }
        it->lastSerial = serial;
        Slot& s = it->slots[frame];

        // The buffer. Reallocated only when the extent changes (a resize, or a
        // view id reused for a different size), and the old one retired rather
        // than destroyed: the frame that last read it is complete, but the
        // retire queue is the rule here and costs nothing.
        if (s.buf.handle == VK_NULL_HANDLE || s.width != ext.width || s.height != ext.height) {
            if (s.buf.handle != VK_NULL_HANDLE) {
                if (retireFn_) retireFn_(std::move(s.buf));
                else destroyBuffer(ctx_.allocator(), s.buf);
                s.buf = Buffer{};
            }
            s.buf = createBuffer(ctx_.allocator(), ctx_.device(),
                                 VkDeviceSize(ext.width) * ext.height * sizeof(float),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                 VMA_MEMORY_USAGE_AUTO, 0);
            if (s.buf.handle == VK_NULL_HANDLE) return false;
            s.width  = ext.width;
            s.height = ext.height;
        }

        // The set: allocated once per (view, frame in flight), rewritten only
        // when the depth view it names is a different object.
        if (s.set == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &dsLayout_;
            if (vkAllocateDescriptorSets(ctx_.device(), &ai, &s.set) != VK_SUCCESS) {
                s.set = VK_NULL_HANDLE;
                return false;
            }
            s.setViewSerial = 0;
        }
        const std::uint64_t viewSerial = imageViewSerial(depthView);
        if (s.setViewSerial != viewSerial) {
            VkDescriptorImageInfo ii{};
            ii.sampler     = sampler_;
            ii.imageView   = depthView;
            ii.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = s.set;
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &ii;
            vulkan::updateDescriptorSets(ctx_.device(), 1, &w, 0, nullptr);
            s.setViewSerial = viewSerial;
        }
        s.preparedSerial = serial;

        out.target       = Target{s.buf.handle, s.buf.address, s.width, s.height};
        out.set          = s.set;
        out.multisampled = multisampled;
        // Column-major: element [col * 4 + row]. viewZ = (A d + B) / (C d + D)
        // with A, B from the z row and C, D from the w row, columns 2 and 3.
        out.lin[0] = projInverse[2 * 4 + 2];
        out.lin[1] = projInverse[3 * 4 + 2];
        out.lin[2] = projInverse[2 * 4 + 3];
        out.lin[3] = projInverse[3 * 4 + 3];
        return true;
    }

    BillboardSoftDepth::Target BillboardSoftDepth::target(std::uint32_t viewId, std::uint32_t frame,
                                                          std::uint64_t serial) const {
        if (frame >= framesInFlight_) return {};
        for (const auto& v : views_) {
            if (v.viewId != viewId) continue;
            const Slot& s = v.slots[frame];
            if (s.preparedSerial != serial || s.buf.handle == VK_NULL_HANDLE) return {};
            return Target{s.buf.handle, s.buf.address, s.width, s.height};
        }
        return {};
    }

    void BillboardSoftDepth::record(VkCommandBuffer cb, const Dispatch& d) const {
        if (d.set == VK_NULL_HANDLE || d.target.address == 0) return;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.multisampled ? pipeMs_ : pipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout_, 0, 1, &d.set, 0, nullptr);
        SoftDepthPc pc{};
        pc.outAddr = d.target.address;
        pc.width   = d.target.width;
        pc.height  = d.target.height;
        std::memcpy(pc.lin, d.lin, sizeof(pc.lin));
        vkCmdPushConstants(cb, pipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (d.target.width + kLocalSize - 1u) / kLocalSize,
                      (d.target.height + kLocalSize - 1u) / kLocalSize, 1);
    }

    void BillboardSoftDepth::declare(rg::RenderGraph& g, rg::PassBuilder& pass, VkImage depth,
                                     const Dispatch& d) const {
        constexpr auto kDsRo = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        if (depth != VK_NULL_HANDLE) {
            pass.use(g.importImage("billboard.softDepthSrc", depth, VK_IMAGE_ASPECT_DEPTH_BIT, 1, kDsRo),
                     rg::sampled(kDsRo, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
        }
        pass.use(g.importBuffer("billboard.softDepth", d.target.buffer),
                 rg::storageWrite(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
    }

}// namespace threepp::vulkan
