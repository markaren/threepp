#include "threepp/renderers/vulkan/WakeFieldPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/wake_field.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/wake_ripple.comp.spv.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace threepp::vulkan {

    static_assert(sizeof(WakeFieldPipeline::PushConstants) == 24,
                  "WakeFieldPipeline::PushConstants must match the Pc of wake_field.comp and wake_ripple.comp (24 bytes)");

    namespace {

        constexpr VkFormat kStateFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        constexpr VkFormat kVelFormat   = VK_FORMAT_R16G16_SFLOAT;
        constexpr VkFormat kAmpFormat   = VK_FORMAT_R32G32B32A32_SFLOAT;
        constexpr VkFormat kWorkFormat  = VK_FORMAT_R32G32_SFLOAT;// what the IFFT's shaders declare
        constexpr VkFormat kSlopeFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        constexpr VkDeviceSize kTableBytes =
                sizeof(WakeTableGpu) + DisplacedMesh::kMaxWakeSources * sizeof(DisplacedMesh::WakeSource);

        constexpr VkImageUsageFlags kUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

        VkImageView viewOf(VulkanContext& ctx, VkImage image, VkFormat format, VkImageViewType type,
                           uint32_t baseMip, uint32_t mips, uint32_t layers, const char* name) {
            VkImageViewCreateInfo vci{};
            vci.sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image            = image;
            vci.viewType         = type;
            vci.format           = format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mips, 0, layers};
            VkImageView view = VK_NULL_HANDLE;
            check(createImageView(ctx.device(), &vci, nullptr, &view), "createImageView(wakeField)");
            ctx.setObjectName(view, name);
            return view;
        }

        void arrayImage(VulkanContext& ctx, VkFormat format, uint32_t res, uint32_t layers, uint32_t mips,
                        const char* name, VkImage& image, VmaAllocation& alloc, VkImageView& view) {
            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = format;
            ici.extent        = {res, res, 1};
            ici.mipLevels     = mips;
            ici.arrayLayers   = layers;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = kUsage;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx.allocator(), &ici, &aci, &image, &alloc, nullptr), "vmaCreateImage(wakeField)");
            ctx.setObjectName(image, name);
            view = viewOf(ctx, image, format, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, mips, layers, name);
        }

        // One rg32f plane as the ocean's IFFT takes them.
        water::OceanImage planeImage(VulkanContext& ctx, uint32_t res, const char* name) {
            water::OceanImage img{};
            img.format = kWorkFormat;
            img.width  = res;
            img.height = res;
            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = kWorkFormat;
            ici.extent        = {res, res, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = kUsage;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(wakeField ripple plane)");
            ctx.setObjectName(img.image, name);
            img.view = viewOf(ctx, img.image, kWorkFormat, VK_IMAGE_VIEW_TYPE_2D, 0, 1, 1, name);
            return img;
        }

        // UNDEFINED -> GENERAL, zeroed (every mip and layer).
        void clearToGeneral(VkCommandBuffer cb, VkImage image) {
            VkImageMemoryBarrier imb{};
            imb.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            imb.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            imb.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
            imb.srcAccessMask       = 0;
            imb.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            imb.image               = image;
            imb.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
            imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &imb);
            const VkClearColorValue cc{};
            vkCmdClearColorImage(cb, image, VK_IMAGE_LAYOUT_GENERAL, &cc, 1, &imb.subresourceRange);
            imb.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            imb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &imb);
        }

        void memoryBarrier(VkCommandBuffer cb, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                           VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
            VkMemoryBarrier mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = srcAccess;
            mb.dstAccessMask = dstAccess;
            vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }

        // Compute writes of one dispatch -> compute reads and writes of the next.
        void computeBarrier(VkCommandBuffer cb) {
            memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }

        VkPipeline computePipeline(VulkanContext& ctx, const uint32_t* spv, size_t bytes, VkPipelineLayout layout,
                                   const char* what) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = bytes;
            smci.pCode    = spv;
            VkShaderModule mod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx.device(), &smci, nullptr, &mod), what);

            VkPipelineShaderStageCreateInfo stage{};
            stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = mod;
            stage.pName  = "main";

            VkComputePipelineCreateInfo cpci{};
            cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage  = stage;
            cpci.layout = layout;
            VkPipeline pipeline = VK_NULL_HANDLE;
            check(vkCreateComputePipelines(ctx.device(), ctx.pipelineCache(), 1, &cpci, nullptr, &pipeline), what);
            vkDestroyShaderModule(ctx.device(), mod, nullptr);
            return pipeline;
        }

    }// namespace

    WakeFieldPipeline::WakeFieldPipeline(VulkanContext& ctx)
        : ctx_(ctx) {
        createPipeline();
    }

    WakeFieldPipeline::~WakeFieldPipeline() {
        VkDevice d = ctx_.device();
        if (dummyView_)            vkDestroyImageView(d, dummyView_, nullptr);
        if (dummyImage_)           vmaDestroyImage(ctx_.allocator(), dummyImage_, dummyAlloc_);
        if (pipeline_)             vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_)       vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)             vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (ripplePipeline_)       vkDestroyPipeline(d, ripplePipeline_, nullptr);
        if (ripplePipelineLayout_) vkDestroyPipelineLayout(d, ripplePipelineLayout_, nullptr);
        if (rippleDsLayout_)       vkDestroyDescriptorSetLayout(d, rippleDsLayout_, nullptr);
        if (descPool_)             vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)              vkDestroySampler(d, sampler_, nullptr);
        if (rippleSampler_)        vkDestroySampler(d, rippleSampler_, nullptr);
    }

    void WakeFieldPipeline::createPipeline() {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(PushConstants);

        // wake_field.comp. Bindings: 0, 1 = state sources, 2 = velocity source
        // (combined image samplers over the array images); 3 = state target,
        // 4 = velocity target (storage images).
        {
            std::array<VkDescriptorSetLayoutBinding, 5> bb{};
            for (uint32_t i = 0; i < 5; ++i) {
                bb[i].binding         = i;
                bb[i].descriptorType  = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bb[i].descriptorCount = 1;
                bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bb.size());
            dlci.pBindings    = bb.data();
            check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
                  "vkCreateDescriptorSetLayout(wakeField)");

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &dsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcr;
            check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
                  "vkCreatePipelineLayout(wakeField)");
            pipeline_ = computePipeline(ctx_, kWakeFieldCompSpv, sizeof(kWakeFieldCompSpv), pipelineLayout_,
                                        "vkCreateComputePipelines(wakeField)");
        }

        // wake_ripple.comp. Bindings: 0 = the spectra (array), 1 = the patch's
        // FFT field, 2 = mip 0 of the slopes (array). All storage images.
        {
            std::array<VkDescriptorSetLayoutBinding, 3> bb{};
            for (uint32_t i = 0; i < 3; ++i) {
                bb[i].binding         = i;
                bb[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bb[i].descriptorCount = 1;
                bb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bb.size());
            dlci.pBindings    = bb.data();
            check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &rippleDsLayout_),
                  "vkCreateDescriptorSetLayout(wakeRipple)");

            VkPipelineLayoutCreateInfo plci{};
            plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount         = 1;
            plci.pSetLayouts            = &rippleDsLayout_;
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges    = &pcr;
            check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &ripplePipelineLayout_),
                  "vkCreatePipelineLayout(wakeRipple)");
            ripplePipeline_ = computePipeline(ctx_, kWakeRippleCompSpv, sizeof(kWakeRippleCompSpv),
                                              ripplePipelineLayout_, "vkCreateComputePipelines(wakeRipple)");
        }

        // An ocean: three sets for the field (one per dispatch of a step) and
        // one a patch for its ripples; freed with its State.
        const uint32_t ripSets = DisplacedMesh::kMaxWakePatches * kMaxOceans;
        std::array<VkDescriptorPoolSize, 2> ps{};
        ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps[0].descriptorCount = 3 * 3 * kMaxOceans;
        ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ps[1].descriptorCount = 2 * 3 * kMaxOceans + 3 * ripSets;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = 3 * kMaxOceans + ripSets;
        dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
        dpci.pPoolSizes    = ps.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(wakeField)");

        // LINEAR, and a zero border: water outside a patch is undisturbed, and
        // that is also what a step reads where the patch has just arrived.
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor  = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_), "vkCreateSampler(wakeField)");

        // The ripples': a period of the patch's sea, so REPEAT; trilinear and
        // anisotropic, because a pixel on water seen along the surface is a
        // long strip.
        VkSamplerCreateInfo rci{};
        rci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        rci.magFilter        = VK_FILTER_LINEAR;
        rci.minFilter        = VK_FILTER_LINEAR;
        rci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        rci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        rci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        rci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        rci.anisotropyEnable = VK_TRUE;
        rci.maxAnisotropy    = 16.0f;
        rci.maxLod           = VK_LOD_CLAMP_NONE;
        check(vkCreateSampler(ctx_.device(), &rci, nullptr, &rippleSampler_), "vkCreateSampler(wakeRipple)");

        arrayImage(ctx_, kStateFormat, 1, 1, 1, "oceanWakeDummy (1x1, bindings 78 and 79 placeholder)",
                   dummyImage_, dummyAlloc_, dummyView_);
    }

    void WakeFieldPipeline::initDummy(VkCommandBuffer cb) {
        clearToGeneral(cb, dummyImage_);
    }

    std::unique_ptr<WakeFieldPipeline::State> WakeFieldPipeline::createState(uint32_t res, uint32_t layers,
                                                                              uint32_t rippleRes) {
        auto st = std::make_unique<State>();
        st->res    = std::clamp(res, 64u, 2048u);
        st->layers = std::clamp(layers, 1u, DisplacedMesh::kMaxWakePatches);
        static const char* const kNames[5] = {"ocean.wake.stateA", "ocean.wake.stateB", "ocean.wake.stateC",
                                              "ocean.wake.velA", "ocean.wake.velB"};
        for (uint32_t i = 0; i < 5; ++i) {
            arrayImage(ctx_, i < 3 ? kStateFormat : kVelFormat, st->res, st->layers, 1, kNames[i],
                       st->image[i], st->alloc[i], st->view[i]);
        }
        st->table = createBuffer(ctx_.allocator(), ctx_.device(), kTableBytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_AUTO);
        ctx_.setObjectName(st->table.handle, "ocean.wake.table");

        std::array<VkDescriptorSetLayout, 3> layouts{dsLayout_, dsLayout_, dsLayout_};
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = descPool_;
        dai.descriptorSetCount = 3;
        dai.pSetLayouts        = layouts.data();
        check(vkAllocateDescriptorSets(ctx_.device(), &dai, st->ds.data()), "vkAllocateDescriptorSets(wakeField)");

        // image index bound at (state0, state1, vel, stateOut, velOut), per dispatch
        static constexpr uint32_t kBind[3][5] = {
                {0, 0, 3, 1, 4},// advect:  A, velA        -> B, velB
                {0, 1, 3, 2, 4},// correct: A, B, velA     -> C   (velB bound, not written)
                {2, 2, 4, 0, 3},// settle:  C, velB        -> A, velA
        };
        for (uint32_t m = 0; m < 3; ++m) {
            std::array<VkDescriptorImageInfo, 5> ii{};
            std::array<VkWriteDescriptorSet, 5>  ws{};
            for (uint32_t b = 0; b < 5; ++b) {
                ii[b].sampler     = b < 3 ? sampler_ : VK_NULL_HANDLE;
                ii[b].imageView   = st->view[kBind[m][b]];
                ii[b].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                ws[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[b].dstSet          = st->ds[m];
                ws[b].dstBinding      = b;
                ws[b].descriptorCount = 1;
                ws[b].descriptorType  = b < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                ws[b].pImageInfo      = &ii[b];
            }
            updateDescriptorSets(ctx_.device(), uint32_t(ws.size()), ws.data(), 0, nullptr);
        }

        if (rippleRes > 0u) {
            uint32_t n = 128u;
            while (n * 2u <= std::min(rippleRes, 1024u)) n *= 2u;
            st->rippleRes = n;
            uint32_t mips = 1u;
            while ((n >> mips) >= 4u) ++mips;// down to 4 texels a side
            st->rippleMips = mips;
            arrayImage(ctx_, kAmpFormat, n, st->layers, 1, "ocean.wake.rippleSpectrum",
                       st->ampImage, st->ampAlloc, st->ampView);
            arrayImage(ctx_, kSlopeFormat, n, st->layers, mips, "ocean.wake.rippleSlopes",
                       st->slopeImage, st->slopeAlloc, st->slopeView);
            st->slopeTarget = viewOf(ctx_, st->slopeImage, kSlopeFormat, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 0, 1, st->layers,
                                     "ocean.wake.rippleSlopes (mip 0)");
            st->scratch = planeImage(ctx_, n, "ocean.wake.rippleScratch");
            st->ifft    = std::make_unique<water::IFFT>(ctx_, n, st->layers);

            std::array<VkDescriptorSetLayout, DisplacedMesh::kMaxWakePatches> rl{};
            rl.fill(rippleDsLayout_);
            VkDescriptorSetAllocateInfo rai{};
            rai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            rai.descriptorPool     = descPool_;
            rai.descriptorSetCount = st->layers;
            rai.pSetLayouts        = rl.data();
            check(vkAllocateDescriptorSets(ctx_.device(), &rai, st->rippleDs.data()),
                  "vkAllocateDescriptorSets(wakeRipple)");
            for (uint32_t i = 0; i < st->layers; ++i) {
                st->work[i] = planeImage(ctx_, n, "ocean.wake.rippleField");
                std::array<VkDescriptorImageInfo, 3> ii{};
                std::array<VkWriteDescriptorSet, 3>  ws{};
                const VkImageView views[3] = {st->ampView, st->work[i].view, st->slopeTarget};
                for (uint32_t b = 0; b < 3; ++b) {
                    ii[b].imageView   = views[b];
                    ii[b].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    ws[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    ws[b].dstSet          = st->rippleDs[i];
                    ws[b].dstBinding      = b;
                    ws[b].descriptorCount = 1;
                    ws[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    ws[b].pImageInfo      = &ii[b];
                }
                updateDescriptorSets(ctx_.device(), uint32_t(ws.size()), ws.data(), 0, nullptr);
            }
        }
        return st;
    }

    void WakeFieldPipeline::initState(VkCommandBuffer cb, State& st) {
        for (uint32_t i = 0; i < 5; ++i) clearToGeneral(cb, st.image[i]);
        vkCmdFillBuffer(cb, st.table.handle, 0, VK_WHOLE_SIZE, 0u);
        if (st.rippleRes > 0u) {
            clearToGeneral(cb, st.ampImage);
            clearToGeneral(cb, st.slopeImage);
            // The IFFT tracks its planes' layouts itself; these are in GENERAL
            // from here on, which is what it wants to find.
            clearToGeneral(cb, st.scratch.image);
            st.scratch.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            for (uint32_t i = 0; i < st.layers; ++i) {
                clearToGeneral(cb, st.work[i].image);
                st.work[i].currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
        }
    }

    void WakeFieldPipeline::destroyState(State& st) {
        VkDevice d = ctx_.device();
        for (uint32_t i = 0; i < 5; ++i) {
            if (st.view[i])  vkDestroyImageView(d, st.view[i], nullptr);
            if (st.image[i]) vmaDestroyImage(ctx_.allocator(), st.image[i], st.alloc[i]);
            st.view[i]  = VK_NULL_HANDLE;
            st.image[i] = VK_NULL_HANDLE;
        }
        if (st.ds[0]) vkFreeDescriptorSets(d, descPool_, uint32_t(st.ds.size()), st.ds.data());
        st.ds = {};
        destroyBuffer(ctx_.allocator(), st.table);

        if (st.rippleRes > 0u) {
            st.ifft.reset();
            if (st.rippleDs[0]) vkFreeDescriptorSets(d, descPool_, st.layers, st.rippleDs.data());
            st.rippleDs = {};
            auto plane = [&](water::OceanImage& img) {
                if (img.view)  vkDestroyImageView(d, img.view, nullptr);
                if (img.image) vmaDestroyImage(ctx_.allocator(), img.image, img.alloc);
                img = {};
            };
            plane(st.scratch);
            for (auto& w : st.work) plane(w);
            if (st.slopeTarget) vkDestroyImageView(d, st.slopeTarget, nullptr);
            if (st.slopeView)   vkDestroyImageView(d, st.slopeView, nullptr);
            if (st.slopeImage)  vmaDestroyImage(ctx_.allocator(), st.slopeImage, st.slopeAlloc);
            if (st.ampView)     vkDestroyImageView(d, st.ampView, nullptr);
            if (st.ampImage)    vmaDestroyImage(ctx_.allocator(), st.ampImage, st.ampAlloc);
            st.slopeTarget = st.slopeView = st.ampView = VK_NULL_HANDLE;
            st.slopeImage = st.ampImage = VK_NULL_HANDLE;
            st.rippleRes = 0;
        }
    }

    void WakeFieldPipeline::record(VkCommandBuffer cb, State& st, const DisplacedMesh& dm, double nowSec) {
        // The first step, and one after the clock went back (a run started
        // over), take no time: they deposit and decay nothing away. Otherwise
        // dt is clamped so a stall cannot throw the field a quarter of a patch
        // away in one step. A frame at the same instant as the last one
        // changes nothing: the table and the images stay as that step left them.
        const double since = nowSec - st.prevTimeSec;
        const bool restart = st.prevTimeSec < 0.0 || since < 0.0;
        const float dt = restart ? 0.0f : std::min(static_cast<float>(since), 0.25f);
        if (!restart && dt <= 0.0f) return;
        st.prevTimeSec = nowSec;

        const DisplacedMesh::WakeField& wf = dm.wakeField;
        struct Upload {
            WakeTableGpu              table;
            DisplacedMesh::WakeSource sources[DisplacedMesh::kMaxWakeSources];
        };
        static_assert(sizeof(Upload) == kTableBytes, "the table and its sources go up in one piece");
        Upload up{};
        WakeTableGpu& tb = up.table;
        tb.patchCount      = st.layers;
        tb.res             = st.res;
        tb.timeSec         = static_cast<float>(std::fmod(nowSec, 4096.0));
        tb.dt              = dt;
        tb.foamLife        = std::max(wf.foamLife, 1e-3f);
        tb.foamThinning    = std::max(wf.foamThinning, 1.0f);
        tb.aerationLife    = std::max(wf.aerationLife, 1e-3f);
        tb.turbulenceLife  = std::max(wf.turbulenceLife, 1e-3f);
        tb.laneLife        = std::max(wf.laneLife, 1e-3f);
        tb.velocityLife    = std::max(wf.velocityLife, 1e-3f);
        tb.eddyTexels      = std::max(wf.eddyTexels, 3.0f);// (a patch's own eddy size is in its record)
        tb.spread          = std::max(wf.spread, 0.0f);
        tb.rippleRes       = st.rippleRes;
        tb.rippleViscosity = std::max(wf.rippleViscosity, 0.0f);
        tb.rippleGain      = wf.rippleGain;

        uint32_t liveMask = 0;
        for (uint32_t i = 0; i < st.layers; ++i) {
            const DisplacedMesh::WakePatch& p = dm.wakePatches[i];
            WakePatchGpu& g = tb.patches[i];
            if (!(p.size > 0.0f)) {
                st.prevSize[i] = 0.0f;// off: whatever the layer held is forgotten
                continue;
            }
            const float texel = p.size / static_cast<float>(st.res);
            g.originX = std::floor((p.centerX - 0.5f * p.size) / texel) * texel;
            g.originZ = std::floor((p.centerZ - 0.5f * p.size) / texel) * texel;
            g.size    = p.size;
            g.texel   = texel;
            // A patch that was off, or has another size, starts from still water.
            const bool history = st.prevSize[i] == p.size;
            g.prevOriginX = history ? st.prevOriginX[i] : g.originX;
            g.prevOriginZ = history ? st.prevOriginZ[i] : g.originZ;
            g.live        = history ? 2.0f : 1.0f;
            g.eddy        = std::max(p.eddy > 0.0f ? p.eddy : wf.eddyTexels * texel, 3.0f * texel);
            st.prevOriginX[i] = g.originX;
            st.prevOriginZ[i] = g.originZ;
            st.prevSize[i]    = p.size;
            // Its ripples' window: its own, or the one it names.
            const bool own = p.rippleSize > 0.0f;
            g.rippleSize    = own ? p.rippleSize : p.size;
            g.rippleOriginX = own ? p.rippleCenterX - 0.5f * p.rippleSize : g.originX;
            g.rippleOriginZ = own ? p.rippleCenterZ - 0.5f * p.rippleSize : g.originZ;
            liveMask |= 1u << i;
        }
        const uint32_t sourceCount = static_cast<uint32_t>(
                std::min<size_t>(dm.wakeSources.size(), DisplacedMesh::kMaxWakeSources));
        if (sourceCount > 0u) {
            std::memcpy(up.sources, dm.wakeSources.data(), sourceCount * sizeof(DisplacedMesh::WakeSource));
        }

        // The ripples: whose patch each pushing source forces (one patch each,
        // because the shade adds the patches' waves), how fast every patch's
        // waves have to die, and which patches hold any waves at all.
        uint32_t rippleMask = 0;
        if (st.rippleRes > 0u) {
            std::array<float, DisplacedMesh::kMaxWakePatches> fastest{};
            std::array<bool, DisplacedMesh::kMaxWakePatches>  pushed{};
            for (uint32_t s = 0; s < sourceCount; ++s) {
                DisplacedMesh::WakeSource& sc = up.sources[s];
                if (sc.push == 0.0f) continue;
                auto holds = [&](uint32_t i) {
                    if (i >= st.layers || !(liveMask & (1u << i))) return false;
                    const WakePatchGpu& g = tb.patches[i];
                    const float u = (sc.x1 - g.rippleOriginX) / g.rippleSize;
                    const float v = (sc.z1 - g.rippleOriginZ) / g.rippleSize;
                    return u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f;
                };
                int32_t owner = -1;
                if (sc.patch >= 0) {
                    if (holds(static_cast<uint32_t>(sc.patch))) owner = sc.patch;
                } else {
                    for (uint32_t i = 0; i < st.layers && owner < 0; ++i) {
                        if (holds(i)) owner = static_cast<int32_t>(i);
                    }
                }
                if (owner < 0) {// in no patch: its waves would come up a period away
                    sc.push = 0.0f;
                    continue;
                }
                sc.patch      = owner;
                pushed[owner] = true;
                if (dt > 0.0f) {
                    fastest[owner] = std::max(fastest[owner], std::hypot(sc.x1 - sc.x0, sc.z1 - sc.z0) / dt);
                }
            }
            const float slowest = 1.0f / std::max(wf.rippleLife, 0.1f);
            for (uint32_t i = 0; i < st.layers; ++i) {
                if (!(liveMask & (1u << i))) {
                    st.rippleHeld[i]  = false;
                    st.rippleQuiet[i] = 0.0f;
                    st.rippleSpeed[i] = 0.0f;
                    continue;
                }
                WakePatchGpu& g = tb.patches[i];
                // another period is another sea: the spectrum it holds means nothing in it
                if (st.rippleSize[i] != g.rippleSize) st.rippleHeld[i] = false;
                st.rippleSize[i] = g.rippleSize;
                // A wake's oldest waves are the transverse ones, which follow at
                // half the boat's speed: a side astern of her they are
                // 2 side / speed old, and 1.5 speed / side leaves e^-3 of them
                // by then. The periodic sea brings back nothing that shows.
                st.rippleSpeed[i] = std::max(fastest[i], st.rippleSpeed[i] * std::exp(-dt / 10.0f));
                g.rippleDecay     = std::max(slowest, 1.5f * st.rippleSpeed[i] / g.rippleSize);
                st.rippleQuiet[i] = pushed[i] ? 0.0f : st.rippleQuiet[i] + dt;
                const bool on = pushed[i] || (st.rippleHeld[i] && st.rippleQuiet[i] * g.rippleDecay < 7.0f);
                if (!on) {
                    st.rippleHeld[i] = false;
                    continue;
                }
                g.ripple         = st.rippleHeld[i] && g.live > 1.5f ? 2.0f : 1.0f;
                st.rippleHeld[i] = true;
                rippleMask |= 1u << i;
            }
        }

        // The table, in the command stream: last frame's water shade may still
        // be reading the old one, and this frame's must read the new.
        VkBufferMemoryBarrier bmb{};
        bmb.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bmb.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        bmb.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.buffer              = st.table.handle;
        bmb.size                = VK_WHOLE_SIZE;
        bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
        vkCmdUpdateBuffer(cb, st.table.handle, 0,
                          sizeof(WakeTableGpu) + sourceCount * sizeof(DisplacedMesh::WakeSource), &up);
        bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &bmb, 0, nullptr);
        if (liveMask == 0u) return;

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const uint32_t groups = (st.res + 7u) / 8u;
        PushConstants pc{};
        pc.tableAddr   = st.table.address;
        pc.sourceCount = sourceCount;
        for (uint32_t mode = 0; mode < 3; ++mode) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &st.ds[mode], 0, nullptr);
            pc.mode = mode;
            for (uint32_t i = 0; i < st.layers; ++i) {
                if (!(liveMask & (1u << i))) continue;
                pc.layer = i;
                vkCmdPushConstants(cb, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cb, groups, groups, 1);
            }
            computeBarrier(cb);
        }

        if (rippleMask != 0u) recordRipples(cb, st, rippleMask, pc);
    }

    void WakeFieldPipeline::recordRipples(VkCommandBuffer cb, State& st, uint32_t rippleMask, PushConstants pc) {
        const uint32_t groups = (st.rippleRes + 7u) / 8u;
        auto dispatch = [&](uint32_t layer, uint32_t mode) {
            // (bound anew each time: the IFFT in between binds its own)
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ripplePipeline_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ripplePipelineLayout_, 0, 1,
                                    &st.rippleDs[layer], 0, nullptr);
            pc.layer = layer;
            pc.mode  = mode;
            vkCmdPushConstants(cb, ripplePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cb, groups, groups, 1);
        };
        for (uint32_t i = 0; i < st.layers; ++i) {
            if (!(rippleMask & (1u << i))) continue;
            dispatch(i, 0u);// turn the spectrum, and force it
            computeBarrier(cb);
            dispatch(i, 1u);// the slopes' spectrum
            computeBarrier(cb);
            st.ifft->recordApply(cb, st.work[i], st.scratch);
            dispatch(i, 2u);// the slopes, into the patch's layer
        }

        // The mips, every layer at once: each a box of the one above, so the
        // third channel stays the mean of the slopes' squares.
        memoryBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        for (uint32_t m = 1; m < st.rippleMips; ++m) {
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1u, 0, st.layers};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, st.layers};
            const int32_t s = static_cast<int32_t>(std::max(st.rippleRes >> (m - 1u), 1u));
            const int32_t t = static_cast<int32_t>(std::max(st.rippleRes >> m, 1u));
            blit.srcOffsets[1] = {s, s, 1};
            blit.dstOffsets[1] = {t, t, 1};
            vkCmdBlitImage(cb, st.slopeImage, VK_IMAGE_LAYOUT_GENERAL, st.slopeImage, VK_IMAGE_LAYOUT_GENERAL,
                           1, &blit, VK_FILTER_LINEAR);
            memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        }
        if (st.rippleMips <= 1u) {
            memoryBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT);
        }
    }

}// namespace threepp::vulkan
