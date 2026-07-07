#include "threepp/renderers/vulkan/DofPass.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/dof_coc.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/dof_combine.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/dof_gather.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/dof_tile.comp.spv.h"

#include <algorithm>
#include <cstring>

namespace threepp::vulkan {

    namespace {

        VkPipeline makeComputePipe(VkDevice d, VkPipelineCache cache, VkPipelineLayout layout,
                                   const uint32_t* spv, size_t spvBytes, const char* label) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = spvBytes;
            smci.pCode    = spv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(d, &smci, nullptr, &mod), label);
            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";
            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = layout;
            VkPipeline pipe = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(d, cache, 1, &cpci, nullptr, &pipe), label);
            vkDestroyShaderModule(d, mod, nullptr);
            return pipe;
        }

    }// namespace

    DofPass::DofPass(VulkanContext& ctx, VkCommandPool cmdPool, uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool), framesInFlight_(framesInFlight) {
        createPipelines();
        createDescriptorPool();
    }

    DofPass::~DofPass() {
        VkDevice d = ctx_.device();
        if (cocPipe_)         vkDestroyPipeline(d, cocPipe_, nullptr);
        if (tilePipe_)        vkDestroyPipeline(d, tilePipe_, nullptr);
        if (gatherPipe_)      vkDestroyPipeline(d, gatherPipe_, nullptr);
        if (combinePipe_)     vkDestroyPipeline(d, combinePipe_, nullptr);
        if (cocPipeLayout_)     vkDestroyPipelineLayout(d, cocPipeLayout_, nullptr);
        if (tilePipeLayout_)    vkDestroyPipelineLayout(d, tilePipeLayout_, nullptr);
        if (gatherPipeLayout_)  vkDestroyPipelineLayout(d, gatherPipeLayout_, nullptr);
        if (combinePipeLayout_) vkDestroyPipelineLayout(d, combinePipeLayout_, nullptr);
        if (cocLayout_)     vkDestroyDescriptorSetLayout(d, cocLayout_, nullptr);
        if (tileLayout_)    vkDestroyDescriptorSetLayout(d, tileLayout_, nullptr);
        if (gatherLayout_)  vkDestroyDescriptorSetLayout(d, gatherLayout_, nullptr);
        if (combineLayout_) vkDestroyDescriptorSetLayout(d, combineLayout_, nullptr);
        if (descPool_)      vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (nearestSampler_) vkDestroySampler(d, nearestSampler_, nullptr);
        if (linearSampler_)  vkDestroySampler(d, linearSampler_, nullptr);
        destroyImages();
    }

    void DofPass::destroyImages() {
        VkDevice d = ctx_.device();
        destroyImage2D(ctx_.allocator(), d, half_);
        destroyImage2D(ctx_.allocator(), d, far_);
        destroyImage2D(ctx_.allocator(), d, near_);
        destroyImage2D(ctx_.allocator(), d, tileA_);
        destroyImage2D(ctx_.allocator(), d, tileB_);
    }

    Image2D DofPass::createImage(uint32_t w, uint32_t h, VkFormat format, const char* label) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = format;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &out.image, &out.alloc, nullptr), label);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view), label);
        ctx_.setObjectName(out.image, label);

        // UNDEFINED → GENERAL once; the pass keeps everything in GENERAL.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(dof)");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(dof)");
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = out.image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        check(vkEndCommandBuffer(cb), "end one-shot cb(dof)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(dof)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(dof)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
        return out;
    }

    void DofPass::createPipelines() {
        VkDevice d = ctx_.device();

        auto makeSampler = [&](VkFilter filter, VkSampler& out) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter    = filter;
            sci.minFilter    = filter;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            check(vkCreateSampler(d, &sci, nullptr, &out), "vkCreateSampler(dof)");
        };
        makeSampler(VK_FILTER_NEAREST, nearestSampler_);
        makeSampler(VK_FILTER_LINEAR, linearSampler_);

        auto makeLayout = [&](std::initializer_list<VkDescriptorType> types,
                              uint32_t pcSize, VkDescriptorSetLayout& dsl,
                              VkPipelineLayout& pl) {
            std::vector<VkDescriptorSetLayoutBinding> bnd;
            uint32_t idx = 0;
            for (VkDescriptorType t : types) {
                VkDescriptorSetLayoutBinding b{};
                b.binding         = idx++;
                b.descriptorType  = t;
                b.descriptorCount = 1;
                b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
                bnd.push_back(b);
            }
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bnd.size());
            dlci.pBindings    = bnd.data();
            check(vkCreateDescriptorSetLayout(d, &dlci, nullptr, &dsl),
                  "vkCreateDescriptorSetLayout(dof)");
            VkPushConstantRange pc{};
            pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pc.offset     = 0;
            pc.size       = pcSize;
            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &dsl;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pc;
            check(vkCreatePipelineLayout(d, &plci, nullptr, &pl), "vkCreatePipelineLayout(dof)");
        };

        constexpr auto S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        constexpr auto I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        constexpr auto U = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        makeLayout({U, S, S, I},    32, cocLayout_, cocPipeLayout_);        // cam, depth, scene → half
        makeLayout({S, I},          20, tileLayout_, tilePipeLayout_);      // src → dst
        makeLayout({S, S, I, I},    16, gatherLayout_, gatherPipeLayout_);  // half, tiles → far, near
        makeLayout({U, S, S, S, I}, 32, combineLayout_, combinePipeLayout_);// cam, depth, far, near → scene

        cocPipe_     = makeComputePipe(d, ctx_.pipelineCache(), cocPipeLayout_, kDofCocCompSpv,
                                       sizeof(kDofCocCompSpv), "vkCreateComputePipelines(dof_coc)");
        tilePipe_    = makeComputePipe(d, ctx_.pipelineCache(), tilePipeLayout_, kDofTileCompSpv,
                                       sizeof(kDofTileCompSpv), "vkCreateComputePipelines(dof_tile)");
        gatherPipe_  = makeComputePipe(d, ctx_.pipelineCache(), gatherPipeLayout_, kDofGatherCompSpv,
                                       sizeof(kDofGatherCompSpv), "vkCreateComputePipelines(dof_gather)");
        combinePipe_ = makeComputePipe(d, ctx_.pipelineCache(), combinePipeLayout_, kDofCombineCompSpv,
                                       sizeof(kDofCombineCompSpv), "vkCreateComputePipelines(dof_combine)");
    }

    void DofPass::createDescriptorPool() {
        // coc(fif) + combine(fif) + tile(2) + gather(1).
        const uint32_t sets = framesInFlight_ * 2 + 3;
        VkDescriptorPoolSize sizes[3]{};
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = framesInFlight_ * 5 + 4;
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = framesInFlight_ * 2 + 4;
        sizes[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sizes[2].descriptorCount = framesInFlight_ * 2;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = sets;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(dof)");

        auto alloc = [&](VkDescriptorSetLayout layout, VkDescriptorSet* out, uint32_t count) {
            std::vector<VkDescriptorSetLayout> layouts(count, layout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = count;
            ai.pSetLayouts        = layouts.data();
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, out),
                  "vkAllocateDescriptorSets(dof)");
        };
        cocSets_.resize(framesInFlight_);
        combineSets_.resize(framesInFlight_);
        alloc(cocLayout_, cocSets_.data(), framesInFlight_);
        alloc(combineLayout_, combineSets_.data(), framesInFlight_);
        alloc(tileLayout_, &tileSet0_, 1);
        alloc(tileLayout_, &tileSet1_, 1);
        alloc(gatherLayout_, &gatherSet_, 1);
    }

    void DofPass::resize(uint32_t width, uint32_t height, const ResizeInputs& in) {
        if (width == 0 || height == 0) return;
        if (width != width_ || height != height_) {
            destroyImages();
            width_  = width;
            height_ = height;
            halfW_  = std::max(width / 2u, 1u);
            halfH_  = std::max(height / 2u, 1u);
            tilesW_ = (halfW_ + 7u) / 8u;
            tilesH_ = (halfH_ + 7u) / 8u;
            half_  = createImage(halfW_, halfH_, VK_FORMAT_R16G16B16A16_SFLOAT, "dof.half");
            far_   = createImage(halfW_, halfH_, VK_FORMAT_R16G16B16A16_SFLOAT, "dof.far");
            near_  = createImage(halfW_, halfH_, VK_FORMAT_R16G16B16A16_SFLOAT, "dof.near");
            tileA_ = createImage(tilesW_, tilesH_, VK_FORMAT_R16G16_SFLOAT, "dof.tileA");
            tileB_ = createImage(tilesW_, tilesH_, VK_FORMAT_R16G16_SFLOAT, "dof.tileB");
        }

        auto sampled = [&](VkImageView v, VkSampler s,
                           VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL) {
            VkDescriptorImageInfo i{};
            i.sampler     = s;
            i.imageView   = v;
            i.imageLayout = layout;
            return i;
        };
        auto storage = [&](VkImageView v) {
            VkDescriptorImageInfo i{};
            i.imageView   = v;
            i.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            return i;
        };

        std::vector<VkWriteDescriptorSet>    writes;
        std::vector<VkDescriptorImageInfo>   imgInfos;
        std::vector<VkDescriptorBufferInfo>  bufInfos;
        imgInfos.reserve(64);
        bufInfos.reserve(8);
        auto wImg = [&](VkDescriptorSet ds, uint32_t bind, VkDescriptorType t,
                        const VkDescriptorImageInfo& info) {
            imgInfos.push_back(info);
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = ds;
            w.dstBinding      = bind;
            w.descriptorCount = 1;
            w.descriptorType  = t;
            w.pImageInfo      = &imgInfos.back();
            writes.push_back(w);
        };
        auto wBuf = [&](VkDescriptorSet ds, uint32_t bind, VkBuffer buf) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = buf;
            bi.range  = VK_WHOLE_SIZE;
            bufInfos.push_back(bi);
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = ds;
            w.dstBinding      = bind;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo     = &bufInfos.back();
            writes.push_back(w);
        };

        constexpr auto S = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        constexpr auto I = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        constexpr VkImageLayout kDepthRO = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            wBuf(cocSets_[f], 0, in.cameraUbos[f]);
            wImg(cocSets_[f], 1, S, sampled(in.depthPerFrame[f], nearestSampler_, kDepthRO));
            wImg(cocSets_[f], 2, S, sampled(in.sceneHdrPerFrame[f], nearestSampler_));
            wImg(cocSets_[f], 3, I, storage(half_.view));

            wBuf(combineSets_[f], 0, in.cameraUbos[f]);
            wImg(combineSets_[f], 1, S, sampled(in.depthPerFrame[f], nearestSampler_, kDepthRO));
            wImg(combineSets_[f], 2, S, sampled(far_.view, linearSampler_));
            wImg(combineSets_[f], 3, S, sampled(near_.view, linearSampler_));
            wImg(combineSets_[f], 4, I, storage(in.sceneHdrPerFrame[f]));
        }
        wImg(tileSet0_, 0, S, sampled(half_.view, nearestSampler_));
        wImg(tileSet0_, 1, I, storage(tileA_.view));
        wImg(tileSet1_, 0, S, sampled(tileA_.view, nearestSampler_));
        wImg(tileSet1_, 1, I, storage(tileB_.view));
        wImg(gatherSet_, 0, S, sampled(half_.view, nearestSampler_));
        wImg(gatherSet_, 1, S, sampled(tileB_.view, nearestSampler_));
        wImg(gatherSet_, 2, I, storage(far_.view));
        wImg(gatherSet_, 3, I, storage(near_.view));

        vkUpdateDescriptorSets(ctx_.device(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    void DofPass::record(VkCommandBuffer cb, uint32_t frame,
                         uint32_t width, uint32_t height,
                         float cocScale, float focusDist, float maxCocPx) {
        const uint32_t hw = std::max(width / 2u, 1u);
        const uint32_t hh = std::max(height / 2u, 1u);
        const uint32_t tw = (hw + 7u) / 8u;
        const uint32_t th = (hh + 7u) / 8u;

        // Global compute→compute barrier. The first one ALSO orders this
        // frame's scratch writes against the previous frame's reads (queue
        // submission-order scope) — the scratch images are shared, not
        // per-frame-in-flight.
        auto barrier = [&]() {
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo di{};
            di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &di);
        };

        struct CocPc {
            uint32_t fullW, fullH, halfW, halfH;
            float cocScale, focusDist, maxCoc, pad;
        };

        // 1. CoC + prefilter → half.
        barrier();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cocPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cocPipeLayout_,
                                0, 1, &cocSets_[frame], 0, nullptr);
        const CocPc cpc{width, height, hw, hh, cocScale, focusDist, maxCocPx, 0.f};
        vkCmdPushConstants(cb, cocPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cpc), &cpc);
        vkCmdDispatch(cb, (hw + 7u) / 8u, (hh + 7u) / 8u, 1);

        // 2. Tile max + dilate.
        barrier();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tilePipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tilePipeLayout_,
                                0, 1, &tileSet0_, 0, nullptr);
        const uint32_t t0pc[5] = {hw, hh, tw, th, 0u};
        vkCmdPushConstants(cb, tilePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(t0pc), t0pc);
        vkCmdDispatch(cb, (tw + 7u) / 8u, (th + 7u) / 8u, 1);
        barrier();
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tilePipeLayout_,
                                0, 1, &tileSet1_, 0, nullptr);
        const uint32_t t1pc[5] = {tw, th, tw, th, 1u};
        vkCmdPushConstants(cb, tilePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(t1pc), t1pc);
        vkCmdDispatch(cb, (tw + 7u) / 8u, (th + 7u) / 8u, 1);

        // 3. Near/far gather.
        barrier();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gatherPipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gatherPipeLayout_,
                                0, 1, &gatherSet_, 0, nullptr);
        const uint32_t gpc[4] = {hw, hh, tw, th};
        vkCmdPushConstants(cb, gatherPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpc), gpc);
        vkCmdDispatch(cb, (hw + 7u) / 8u, (hh + 7u) / 8u, 1);

        // 4. Full-res combine → sceneHdr (RMW). The downstream bloom /
        // PostComposite leading barriers make this write visible.
        barrier();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, combinePipe_);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, combinePipeLayout_,
                                0, 1, &combineSets_[frame], 0, nullptr);
        const CocPc mpc{width, height, hw, hh, cocScale, focusDist, maxCocPx, 0.f};
        vkCmdPushConstants(cb, combinePipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mpc), &mpc);
        vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    }

}// namespace threepp::vulkan
