#include "threepp/renderers/vulkan/TaaResolve.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/taa_resolve.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/rcas.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/motion_tilemax.comp.spv.h"
#include "threepp/renderers/vulkan/shaders/motion_blur.comp.spv.h"

#include <array>
#include <cstring>

namespace threepp::vulkan {

    TaaResolve::TaaResolve(VulkanContext& ctx,
                           VkCommandPool cmdPool,
                           uint32_t imageCount,
                           uint32_t framesInFlight)
        : ctx_(ctx), cmdPool_(cmdPool),
          imageCount_(imageCount), framesInFlight_(framesInFlight) {
        inputImagesPP_.resize(framesInFlight_);
        tileMax_.resize(framesInFlight_);
        mblurOut_.resize(framesInFlight_);
        createPipeline();
        createDescriptorPool();
    }

    TaaResolve::~TaaResolve() {
        VkDevice d = ctx_.device();
        if (pipeline_)       vkDestroyPipeline(d, pipeline_, nullptr);
        if (pipelineLayout_) vkDestroyPipelineLayout(d, pipelineLayout_, nullptr);
        if (dsLayout_)       vkDestroyDescriptorSetLayout(d, dsLayout_, nullptr);
        if (rcasPipe_)       vkDestroyPipeline(d, rcasPipe_, nullptr);
        if (rcasPipeLayout_) vkDestroyPipelineLayout(d, rcasPipeLayout_, nullptr);
        if (rcasDsLayout_)   vkDestroyDescriptorSetLayout(d, rcasDsLayout_, nullptr);
        if (tilemaxPipe_)       vkDestroyPipeline(d, tilemaxPipe_, nullptr);
        if (tilemaxPipeLayout_) vkDestroyPipelineLayout(d, tilemaxPipeLayout_, nullptr);
        if (mblurPipe_)         vkDestroyPipeline(d, mblurPipe_, nullptr);
        if (mblurPipeLayout_)   vkDestroyPipelineLayout(d, mblurPipeLayout_, nullptr);
        if (mblurDsLayout_)     vkDestroyDescriptorSetLayout(d, mblurDsLayout_, nullptr);
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (sampler_)        vkDestroySampler(d, sampler_, nullptr);
        destroyImages();
    }

    void TaaResolve::destroyImages() {
        VkDevice d = ctx_.device();
        for (auto& img : inputImagesPP_)   destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : historyImagesPP_) destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : tileMax_)         destroyImage2D(ctx_.allocator(), d, img);
        for (auto& img : mblurOut_)        destroyImage2D(ctx_.allocator(), d, img);
        historyValid_ = false;
        historyInvalidResolves_ = 2;
    }

    Image2D TaaResolve::createStorageSampledImage(uint32_t w, uint32_t h,
                                                  VkFormat format,
                                                  const char* label) {
        Image2D out{};
        out.width  = w;
        out.height = h;
        out.format = format;

        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = out.format;
        ici.extent        = {w, h, 1};
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_DST because these are clearable scratch targets, same rule
        // the deferred G-buffer storage images follow. It is not optional for
        // taa.history: that slot is what DLSS/FSR are handed as their output
        // resource, and NGX clears it internally on OUR command buffer — which
        // is a spec violation (VUID-vkCmdClearColorImage-image-00002) unless
        // the usage bit is here, since the clear is issued by the SDK and can't
        // be routed through a render-pass loadOp on our side.
        // TRANSFER_SRC for the determinism audit's host readback
        // (VulkanRenderer::readTaaDebugImages) — a usage bit is free.
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &out.image, &out.alloc, nullptr),
              label);

        transitionFreshImage(out.image);

        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = out.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = out.format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        check(vkCreateImageView(ctx_.device(), &vci, nullptr, &out.view),
              "vkCreateImageView(taa)");
        ctx_.setObjectName(out.image, label);
        ctx_.setObjectName(out.view,  label);
        return out;
    }

    void TaaResolve::transitionFreshImage(VkImage img) {
        // One-shot UNDEFINED → GENERAL so the first frame's storage + sampled
        // accesses work without further layout management.
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb),
              "alloc one-shot cb(taa)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(taa)");

        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(taa)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE),
              "submit one-shot(taa)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(taa)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void TaaResolve::createImages(uint32_t inWidth, uint32_t inHeight,
                                  uint32_t outWidth, uint32_t outHeight) {
        destroyImages();
        // Input: BGRA8_UNORM at the render extent — matches denoise.comp's
        // rgba8 output and the swapchain channel order.
        for (auto& img : inputImagesPP_)
            img = createStorageSampledImage(inWidth, inHeight,
                                            VK_FORMAT_B8G8R8A8_UNORM,
                                            "vmaCreateImage(taa.input)");
        // History: RGBA16F at the output extent — the running mix() stays
        // sub-quantum precise and the reconstructed full-res image
        // accumulates here when the input is lower-resolution.
        for (auto& img : historyImagesPP_)
            img = createStorageSampledImage(outWidth, outHeight,
                                            VK_FORMAT_R16G16B16A16_SFLOAT,
                                            "vmaCreateImage(taa.history)");
        // Motion blur: per-32px-tile dominant velocity + (when RCAS is also
        // active) the pre-sharpen blurred frame. Both at the OUTPUT extent —
        // the blur runs on the resolved full-res image.
        tilesX_ = (outWidth  + 31u) / 32u;
        tilesY_ = (outHeight + 31u) / 32u;
        for (auto& img : tileMax_)
            img = createStorageSampledImage(tilesX_, tilesY_,
                                            VK_FORMAT_R16G16_SFLOAT,
                                            "vmaCreateImage(taa.mblurTileMax)");
        for (auto& img : mblurOut_)
            img = createStorageSampledImage(outWidth, outHeight,
                                            VK_FORMAT_B8G8R8A8_UNORM,
                                            "vmaCreateImage(taa.mblurOut)");
    }

    void TaaResolve::createPipeline() {
        if (sampler_ == VK_NULL_HANDLE) {
            VkSamplerCreateInfo sci{};
            sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            // LINEAR filter — required by the Catmull-Rom 5-tap history
            // reconstruction in taa_resolve.comp (each tap fuses 4 texels
            // via the bilinear sampler). A naive single bilinear sample
            // would compound a half-pixel blur every frame on translating
            // close objects — the long-standing "everything smears" bug.
            sci.magFilter    = VK_FILTER_LINEAR;
            sci.minFilter    = VK_FILTER_LINEAR;
            sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod       = 0.f;
            check(vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_),
                  "vkCreateSampler(taa)");
        }
        // Descriptor set layout — 8 bindings:
        //   0..2: combined image samplers — taaInput, historyRead, gbufMotion
        //   3..4: storage images          — swapOut, historyWrite
        //   5..6: combined image samplers — gbufIds (curr + prev)
        //   7:    combined image sampler  — prev-frame depth
        VkDescriptorSetLayoutBinding bindings[8]{};
        for (int i = 0; i < 3; ++i) {
            bindings[i].binding         = i;
            bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[3].binding         = 3;
        bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[4].binding         = 4;
        bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[5].binding         = 5;
        bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[6].binding         = 6;
        bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[6].descriptorCount = 1;
        bindings[6].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        // Binding 7: prev-frame depth (sampled) for depth-based disocclusion.
        bindings[7].binding         = 7;
        bindings[7].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = 8;
        dlci.pBindings    = bindings;
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &dsLayout_),
              "vkCreateDescriptorSetLayout(taa)");

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 128;// scalars + dstOffset(@28) + mat4 skyReproj(@32) + phys dims(@96) + depthLin(@112)

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &dsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &pipelineLayout_),
              "vkCreatePipelineLayout(taa)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kTaaResolveCompSpv);
        smci.pCode    = kTaaResolveCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(taa)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = pipelineLayout_;
        check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(),
                                       1, &cpci, nullptr, &pipeline_),
              "vkCreateComputePipelines(taa)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);

        // ── RCAS sharpen pipeline: sampled resolved @0, storage swapchain @1;
        //    16-byte PC (width, height, amount, pad). ──────────────────────
        {
            VkDescriptorSetLayoutBinding rb[2]{};
            rb[0].binding = 0;
            rb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            rb[0].descriptorCount = 1;
            rb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            rb[1].binding = 1;
            rb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            rb[1].descriptorCount = 1;
            rb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo rdlci{};
            rdlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            rdlci.bindingCount = 2;
            rdlci.pBindings    = rb;
            check(vkCreateDescriptorSetLayout(ctx_.device(), &rdlci, nullptr, &rcasDsLayout_),
                  "vkCreateDescriptorSetLayout(rcas)");

            VkPushConstantRange rpc{};
            rpc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            rpc.offset     = 0;
            rpc.size       = 16;
            VkPipelineLayoutCreateInfo rplci{};
            rplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            rplci.setLayoutCount         = 1;
            rplci.pSetLayouts            = &rcasDsLayout_;
            rplci.pushConstantRangeCount = 1;
            rplci.pPushConstantRanges    = &rpc;
            check(vkCreatePipelineLayout(ctx_.device(), &rplci, nullptr, &rcasPipeLayout_),
                  "vkCreatePipelineLayout(rcas)");

            VkShaderModuleCreateInfo rsmci{};
            rsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            rsmci.codeSize = sizeof(kRcasCompSpv);
            rsmci.pCode    = kRcasCompSpv;
            VkShaderModule rmod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &rsmci, nullptr, &rmod),
                  "vkCreateShaderModule(rcas)");
            VkPipelineShaderStageCreateInfo rstage{};
            rstage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            rstage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            rstage.module = rmod;
            rstage.pName  = "main";
            VkComputePipelineCreateInfo rcpci{};
            rcpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            rcpci.stage  = rstage;
            rcpci.layout = rcasPipeLayout_;
            check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &rcpci,
                                           nullptr, &rcasPipe_),
                  "vkCreateComputePipelines(rcas)");
            vkDestroyShaderModule(ctx_.device(), rmod, nullptr);
        }

        // ── Motion blur pipelines (McGuire 2012). TileMax shares the RCAS
        //    descriptor shape (sampler @0 + storage @1) but needs a 96-byte
        //    PC (scalars + skyReproj mat4); reconstruction gets its own
        //    4-binding layout + 112-byte PC (adds depthLin). ───────────────
        {
            VkPushConstantRange tpc{};
            tpc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            tpc.offset     = 0;
            tpc.size       = 96;// 8 scalars + mat4 skyReproj @32
            VkPipelineLayoutCreateInfo tplci{};
            tplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            tplci.setLayoutCount         = 1;
            tplci.pSetLayouts            = &rcasDsLayout_;
            tplci.pushConstantRangeCount = 1;
            tplci.pPushConstantRanges    = &tpc;
            check(vkCreatePipelineLayout(ctx_.device(), &tplci, nullptr, &tilemaxPipeLayout_),
                  "vkCreatePipelineLayout(mblur.tilemax)");

            VkShaderModuleCreateInfo tsmci{};
            tsmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            tsmci.codeSize = sizeof(kMotionTilemaxCompSpv);
            tsmci.pCode    = kMotionTilemaxCompSpv;
            VkShaderModule tmod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &tsmci, nullptr, &tmod),
                  "vkCreateShaderModule(mblur.tilemax)");
            VkPipelineShaderStageCreateInfo tstage{};
            tstage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            tstage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            tstage.module = tmod;
            tstage.pName  = "main";
            VkComputePipelineCreateInfo tcpci{};
            tcpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            tcpci.stage  = tstage;
            tcpci.layout = tilemaxPipeLayout_;
            check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &tcpci,
                                           nullptr, &tilemaxPipe_),
                  "vkCreateComputePipelines(mblur.tilemax)");
            vkDestroyShaderModule(ctx_.device(), tmod, nullptr);

            VkDescriptorSetLayoutBinding mb[4]{};
            for (int i = 0; i < 3; ++i) {
                mb[i].binding         = static_cast<uint32_t>(i);
                mb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                mb[i].descriptorCount = 1;
                mb[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            mb[3].binding         = 3;
            mb[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            mb[3].descriptorCount = 1;
            mb[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo mdlci{};
            mdlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            mdlci.bindingCount = 4;
            mdlci.pBindings    = mb;
            check(vkCreateDescriptorSetLayout(ctx_.device(), &mdlci, nullptr, &mblurDsLayout_),
                  "vkCreateDescriptorSetLayout(mblur)");

            VkPushConstantRange mpc{};
            mpc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            mpc.offset     = 0;
            mpc.size       = 112;// 8 scalars + mat4 skyReproj @32 + vec4 depthLin @96
            VkPipelineLayoutCreateInfo mplci{};
            mplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            mplci.setLayoutCount         = 1;
            mplci.pSetLayouts            = &mblurDsLayout_;
            mplci.pushConstantRangeCount = 1;
            mplci.pPushConstantRanges    = &mpc;
            check(vkCreatePipelineLayout(ctx_.device(), &mplci, nullptr, &mblurPipeLayout_),
                  "vkCreatePipelineLayout(mblur)");

            VkShaderModuleCreateInfo msmci{};
            msmci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            msmci.codeSize = sizeof(kMotionBlurCompSpv);
            msmci.pCode    = kMotionBlurCompSpv;
            VkShaderModule mmod = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &msmci, nullptr, &mmod),
                  "vkCreateShaderModule(mblur)");
            VkPipelineShaderStageCreateInfo mstage{};
            mstage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            mstage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            mstage.module = mmod;
            mstage.pName  = "main";
            VkComputePipelineCreateInfo mcpci{};
            mcpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            mcpci.stage  = mstage;
            mcpci.layout = mblurPipeLayout_;
            check(vkCreateComputePipelines(ctx_.device(), ctx_.pipelineCache(), 1, &mcpci,
                                           nullptr, &mblurPipe_),
                  "vkCreateComputePipelines(mblur)");
            vkDestroyShaderModule(ctx_.device(), mmod, nullptr);
        }
    }

    void TaaResolve::createDescriptorPool() {
        const uint32_t totalSets = imageCount_ * framesInFlight_;
        VkDescriptorPoolSize sizes[2]{};
        // Main resolve set: 6 sampled + 2 storage. RCAS set: 1 sampled + 1
        // storage (× 2 families: history-slot RCAS + FSR/DLSS postFinalize
        // RCAS). Motion blur: tilemax (1+1) + mblur→out (3+1) per frame,
        // mblur→swap (3+1) + rcas-from-mblur (1+1) per frame×image. All
        // families share this pool.
        sizes[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sizes[0].descriptorCount = totalSets * (6 + 1 + 3 + 1 + 1) + framesInFlight_ * (1 + 3);
        sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sizes[1].descriptorCount = totalSets * (2 + 1 + 1 + 1 + 1) + framesInFlight_ * (1 + 1);

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = totalSets * 5 + framesInFlight_ * 2;// main + rcas + mblurSwap + rcasMb + postFinalizeRcas + tilemax + mblurOut
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes    = sizes;
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(taa)");

        auto allocSets = [&](VkDescriptorSetLayout layout, uint32_t count,
                             std::vector<VkDescriptorSet>& out, const char* what) {
            std::vector<VkDescriptorSetLayout> layouts(count, layout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = descPool_;
            ai.descriptorSetCount = count;
            ai.pSetLayouts        = layouts.data();
            out.resize(count);
            check(vkAllocateDescriptorSets(ctx_.device(), &ai, out.data()), what);
        };
        allocSets(dsLayout_,      totalSets,       descSets_,      "vkAllocateDescriptorSets(taa)");
        allocSets(rcasDsLayout_,  totalSets,       rcasSets_,      "vkAllocateDescriptorSets(rcas)");
        allocSets(rcasDsLayout_,  framesInFlight_, tilemaxSets_,   "vkAllocateDescriptorSets(mblur.tilemax)");
        allocSets(mblurDsLayout_, totalSets,       mblurSwapSets_, "vkAllocateDescriptorSets(mblur.swap)");
        allocSets(mblurDsLayout_, framesInFlight_, mblurOutSets_,  "vkAllocateDescriptorSets(mblur.out)");
        allocSets(rcasDsLayout_,  totalSets,       rcasMbSets_,    "vkAllocateDescriptorSets(rcas.mblur)");
        allocSets(rcasDsLayout_,  totalSets,       postFinalizeRcasSets_, "vkAllocateDescriptorSets(rcas.postFinalize)");
    }

    void TaaResolve::rewriteDescriptors(const DescriptorWriteInputs& inputs) {
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            for (uint32_t i = 0; i < imageCount_; ++i) {
                const uint32_t idx = f * imageCount_ + i;
                const uint32_t readSlot  = 1u - (f & 1u);
                const uint32_t writeSlot = (f & 1u);

                VkDescriptorImageInfo inputI{};
                inputI.sampler     = sampler_;
                inputI.imageView   = inputImagesPP_[f].view;
                inputI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo histReadI{};
                histReadI.sampler     = sampler_;
                histReadI.imageView   = historyImagesPP_[readSlot].view;
                histReadI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo motionI{};
                motionI.sampler     = inputs.gbufSampler;
                motionI.imageView   = inputs.gbufMotionPerFrame[f];
                motionI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkDescriptorImageInfo swapI{};
                swapI.imageView   = inputs.swapchainViews[i];
                swapI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                VkDescriptorImageInfo histWriteI{};
                histWriteI.imageView   = historyImagesPP_[writeSlot].view;
                histWriteI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                // Curr / prev gbuffer IDs for mesh-ID rejection + skinned
                // detection. Prev gbuffer is the OTHER frame-in-flight slot.
                const uint32_t prevFrame = (f + (framesInFlight_ - 1u)) % framesInFlight_;
                VkDescriptorImageInfo idsCurrI{};
                idsCurrI.sampler     = inputs.gbufSampler;
                idsCurrI.imageView   = inputs.gbufIdsPerFrame[f];
                idsCurrI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkDescriptorImageInfo idsPrevI{};
                idsPrevI.sampler     = inputs.gbufSampler;
                idsPrevI.imageView   = inputs.gbufIdsPerFrame[prevFrame];
                idsPrevI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                // Binding 7: PREV-frame depth (the other fif slot), sampled at the
                // reprojected UV for depth-based disocclusion. DEPTH_STENCIL_READ_ONLY
                // layout — the same image+layout the deferred shade pass already reads.
                // Fallback to the motion view (type-safe float sampler2D) if no depth
                // source is supplied; the shader's depthLin is then zero → never sampled.
                VkDescriptorImageInfo depthPrevI{};
                depthPrevI.sampler     = inputs.gbufSampler;
                depthPrevI.imageView   = inputs.gbufDepthPerFrame
                                             ? inputs.gbufDepthPerFrame[prevFrame]
                                             : inputs.gbufMotionPerFrame[prevFrame];
                depthPrevI.imageLayout = inputs.gbufDepthPerFrame
                                             ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                             : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet w[8]{};
                for (int b = 0; b < 8; ++b) {
                    w[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[b].dstSet          = descSets_[idx];
                    w[b].dstBinding      = static_cast<uint32_t>(b);
                    w[b].descriptorCount = 1;
                }
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[0].pImageInfo = &inputI;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[1].pImageInfo = &histReadI;
                w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[2].pImageInfo = &motionI;
                w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[3].pImageInfo = &swapI;
                w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[4].pImageInfo = &histWriteI;
                w[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[5].pImageInfo = &idsCurrI;
                w[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[6].pImageInfo = &idsPrevI;
                w[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[7].pImageInfo = &depthPrevI;
                vkUpdateDescriptorSets(ctx_.device(), 8, w, 0, nullptr);

                // RCAS set: this frame's resolved output lives in the history
                // WRITE slot (writeSlot) → sample it, sharpen, write swapchain.
                VkDescriptorImageInfo rcasIn{};
                rcasIn.sampler     = sampler_;
                rcasIn.imageView   = historyImagesPP_[writeSlot].view;
                rcasIn.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo rcasOut{};
                rcasOut.imageView   = inputs.swapchainViews[i];
                rcasOut.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet rw[2]{};
                rw[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                rw[0].dstSet          = rcasSets_[idx];
                rw[0].dstBinding      = 0;
                rw[0].descriptorCount = 1;
                rw[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                rw[0].pImageInfo      = &rcasIn;
                rw[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                rw[1].dstSet          = rcasSets_[idx];
                rw[1].dstBinding      = 1;
                rw[1].descriptorCount = 1;
                rw[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                rw[1].pImageInfo      = &rcasOut;
                vkUpdateDescriptorSets(ctx_.device(), 2, rw, 0, nullptr);

                // Motion-blur reconstruction → swapchain (the sharpen-off
                // chain): resolved history + motion + tileMax in, swapchain
                // out. And the RCAS variant that reads the blurred frame
                // (mblurOut) instead of the history slot (sharpen-on chain).
                VkDescriptorImageInfo mbColorI{};
                mbColorI.sampler     = sampler_;
                mbColorI.imageView   = historyImagesPP_[writeSlot].view;
                mbColorI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo mbTileI{};
                mbTileI.sampler     = sampler_;
                mbTileI.imageView   = tileMax_[f].view;
                mbTileI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet mw[4]{};
                for (int b = 0; b < 4; ++b) {
                    mw[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    mw[b].dstSet          = mblurSwapSets_[idx];
                    mw[b].dstBinding      = static_cast<uint32_t>(b);
                    mw[b].descriptorCount = 1;
                    mw[b].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                }
                mw[0].pImageInfo     = &mbColorI;
                mw[1].pImageInfo     = &motionI;
                mw[2].pImageInfo     = &mbTileI;
                mw[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                mw[3].pImageInfo     = &swapI;
                vkUpdateDescriptorSets(ctx_.device(), 4, mw, 0, nullptr);

                VkDescriptorImageInfo mbOutReadI{};
                mbOutReadI.sampler     = sampler_;
                mbOutReadI.imageView   = mblurOut_[f].view;
                mbOutReadI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet rmw[2]{};
                for (int b = 0; b < 2; ++b) {
                    rmw[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    rmw[b].dstSet          = rcasMbSets_[idx];
                    rmw[b].dstBinding      = static_cast<uint32_t>(b);
                    rmw[b].descriptorCount = 1;
                }
                rmw[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                rmw[0].pImageInfo     = &mbOutReadI;
                rmw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                rmw[1].pImageInfo     = &rcasOut;
                vkUpdateDescriptorSets(ctx_.device(), 2, rmw, 0, nullptr);
            }

            // Per-frame (swapchain-image-independent) motion-blur sets:
            // TileMax (motion in, tile velocities out) and the reconstruction
            // variant that writes mblurOut for a downstream RCAS.
            {
                const uint32_t writeSlot = (f & 1u);
                VkDescriptorImageInfo motionI{};
                motionI.sampler     = inputs.gbufSampler;
                motionI.imageView   = inputs.gbufMotionPerFrame[f];
                motionI.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkDescriptorImageInfo tileWriteI{};
                tileWriteI.imageView   = tileMax_[f].view;
                tileWriteI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet tw[2]{};
                for (int b = 0; b < 2; ++b) {
                    tw[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    tw[b].dstSet          = tilemaxSets_[f];
                    tw[b].dstBinding      = static_cast<uint32_t>(b);
                    tw[b].descriptorCount = 1;
                }
                tw[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                tw[0].pImageInfo     = &motionI;
                tw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                tw[1].pImageInfo     = &tileWriteI;
                vkUpdateDescriptorSets(ctx_.device(), 2, tw, 0, nullptr);

                VkDescriptorImageInfo mbColorI{};
                mbColorI.sampler     = sampler_;
                mbColorI.imageView   = historyImagesPP_[writeSlot].view;
                mbColorI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo mbTileI{};
                mbTileI.sampler     = sampler_;
                mbTileI.imageView   = tileMax_[f].view;
                mbTileI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo mbOutWriteI{};
                mbOutWriteI.imageView   = mblurOut_[f].view;
                mbOutWriteI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet mw[4]{};
                for (int b = 0; b < 4; ++b) {
                    mw[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    mw[b].dstSet          = mblurOutSets_[f];
                    mw[b].dstBinding      = static_cast<uint32_t>(b);
                    mw[b].descriptorCount = 1;
                    mw[b].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                }
                mw[0].pImageInfo     = &mbColorI;
                mw[1].pImageInfo     = &motionI;
                mw[2].pImageInfo     = &mbTileI;
                mw[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                mw[3].pImageInfo     = &mbOutWriteI;
                vkUpdateDescriptorSets(ctx_.device(), 4, mw, 0, nullptr);
            }
        }
    }

    void TaaResolve::recordResolve(VkCommandBuffer cb,
                                   uint32_t frame,
                                   uint32_t imageIndex,
                                   uint32_t inWidth,
                                   uint32_t inHeight,
                                   uint32_t outWidth,
                                   uint32_t outHeight,
                                   float blendAlpha,
                                   float dtFrames,
                                   bool sharpen,
                                   float sharpenAmount,
                                   const float* skyReproj,
                                   uint32_t dstX,
                                   uint32_t dstY,
                                   uint32_t physInW,
                                   uint32_t physInH,
                                   uint32_t physOutW,
                                   uint32_t physOutH,
                                   const float* depthLin,
                                   float mblurShutter,
                                   float jitterTexX,
                                   float jitterTexY) {
        // Physical (full texture) sizes default to the region sizes → scale 1.
        if (physInW == 0)  physInW  = inWidth;
        if (physInH == 0)  physInH  = inHeight;
        if (physOutW == 0) physOutW = outWidth;
        if (physOutH == 0) physOutH = outHeight;
        const bool sharpenHere = sharpen;
        // Motion blur is full-frame only: split-screen panes (offset stores /
        // region-sized content in full-size textures) skip it silently.
        const bool mblur = mblurShutter > 0.f &&
                           dstX == 0 && dstY == 0 &&
                           physOutW == outWidth && physOutH == outHeight &&
                           tileMax_[frame].view != VK_NULL_HANDLE;
        // Barrier: taaInput write → read; both history slots covered (RAW
        // hazard on the read slot, WAW on the write slot we're about to
        // overwrite this frame).
        std::array<VkImageMemoryBarrier2, 3> pre{};
        for (auto& b : pre) {
            b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                              VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                              VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            b.subresourceRange.levelCount = 1;
            b.subresourceRange.layerCount = 1;
        }
        pre[0].image = inputImagesPP_[frame].image;
        pre[1].image = historyImagesPP_[0].image;
        pre[2].image = historyImagesPP_[1].image;
        VkDependencyInfo dep{};
        dep.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount  = static_cast<uint32_t>(pre.size());
        dep.pImageMemoryBarriers     = pre.data();
        vkCmdPipelineBarrier2(cb, &dep);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        const uint32_t descIdx = frame * imageCount_ + imageIndex;
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout_, 0, 1,
                                &descSets_[descIdx], 0, nullptr);
        // First-frame history is undefined → force alpha=1 so the resolved
        // output is purely the current frame and garbage doesn't bleed into
        // a permanent history. Subsequent frames use the caller's alpha.
        const float alpha = (historyInvalidResolves_ > 0) ? 1.0f : blendAlpha;
        uint32_t alphaBits;
        std::memcpy(&alphaBits, &alpha, sizeof(alphaBits));
        // Layout: blendAlpha, output w/h (history + dispatch + writes),
        // input w/h (the render extent the samples were traced at).
        // Layout mirrors the shader's std430 push block: 7 scalars, 4 bytes
        // of pad, then the column-major mat4 at offset 32. physIn/physOut are
        // each packed into ONE uint (w|h<<16, like dstOffset). Offsets 104/108
        // carry this frame's jitter (jitterTexX/Y — the slots double as the
        // padding that keeps depthLin 16-byte-aligned at offset 112).
        float pc[32] = {};
        std::memcpy(&pc[0], &alphaBits, 4);
        const uint32_t dims[4] = {outWidth, outHeight, inWidth, inHeight};
        std::memcpy(&pc[1], dims, 16);
        // bit0 = writeSwap (this pass writes the swapchain directly — only
        // when nothing downstream needs the history/swap split).
        const uint32_t writeSwapBit = (sharpenHere || mblur) ? 0u : 1u;
        const uint32_t flags = writeSwapBit;
        std::memcpy(&pc[5], &flags, 4);
        pc[6] = dtFrames;
        const uint32_t packedDst = (dstX & 0xFFFFu) | (dstY << 16);
        std::memcpy(&pc[7], &packedDst, 4);// offset 28: swapchain write offset
        std::memcpy(&pc[8], skyReproj, 64);// offset 32: mat4 (verbatim — never patched)
        const uint32_t packedPhysIn  = (physInW  & 0xFFFFu) | (physInH  << 16);
        const uint32_t packedPhysOut = (physOutW & 0xFFFFu) | (physOutH << 16);
        std::memcpy(&pc[24], &packedPhysIn,  4);// offset 96
        std::memcpy(&pc[25], &packedPhysOut, 4);// offset 100
        // This frame's jitter (RENDER TEXELS) in its own slots (offsets
        // 104/108) — the resolve's jitter cancellation reads pc.jitterTexX/Y.
        pc[26] = jitterTexX;
        pc[27] = jitterTexY;
        // offset 112: reverse-Z viewZ linearization (A,B,C,D) for depth
        // disocclusion. Null ⇒ leave zero ⇒ shader disables the gate.
        if (depthLin) std::memcpy(&pc[28], depthLin, 16);
        vkCmdPushConstants(cb, pipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), pc);
        // Dispatch covers the output extent — one thread per full-res pixel.
        const uint32_t gx = (outWidth  + 7u) / 8u;
        const uint32_t gy = (outHeight + 7u) / 8u;
        vkCmdDispatch(cb, gx, gy, 1);

        // Reused by every downstream stage: all producers/consumers here are
        // compute storage accesses on GENERAL-layout images.
        auto computeBarrier = [&] {
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

        if (mblur) {
            // TileMax: dominant velocity per 32px tile. Independent of the
            // resolve's output (reads only the G-buffer motion), so no
            // barrier before it — the GPU may overlap the two dispatches.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, tilemaxPipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    tilemaxPipeLayout_, 0, 1,
                                    &tilemaxSets_[frame], 0, nullptr);
            float tpc[24] = {};
            const uint32_t tdims[4] = {outWidth, outHeight, tilesX_, tilesY_};
            std::memcpy(&tpc[0], tdims, 16);
            tpc[4] = mblurShutter;
            std::memcpy(&tpc[8], skyReproj, 64);// mat4 @32
            vkCmdPushConstants(cb, tilemaxPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(tpc), tpc);
            vkCmdDispatch(cb, tilesX_, tilesY_, 1);// one workgroup per tile

            // Resolve's history write + tilemax's velocity write → the
            // reconstruction's reads.
            computeBarrier();

            // Reconstruction: gathers the resolved frame along the tile
            // neighborhood's dominant velocity. Writes the swapchain, or the
            // pre-sharpen intermediate when RCAS runs after us.
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, mblurPipe_);
            const VkDescriptorSet* mblurTargetSet =
                    sharpenHere ? &mblurOutSets_[frame]
                                : &mblurSwapSets_[descIdx];
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    mblurPipeLayout_, 0, 1,
                                    mblurTargetSet, 0, nullptr);
            float mpc[28] = {};
            std::memcpy(&mpc[0], tdims, 16);
            mpc[4] = mblurShutter;
            std::memcpy(&mpc[8], skyReproj, 64);// mat4 @32
            if (depthLin) std::memcpy(&mpc[24], depthLin, 16);// vec4 @96
            vkCmdPushConstants(cb, mblurPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(mpc), mpc);
            vkCmdDispatch(cb, gx, gy, 1);
        }

        if (sharpenHere) {
            // Make the upstream write (resolve's history slot, or the motion
            // blur's intermediate) visible, then RCAS-sharpen it into the
            // swapchain.
            computeBarrier();

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rcasPipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rcasPipeLayout_, 0, 1,
                                    mblur ? &rcasMbSets_[descIdx]
                                          : &rcasSets_[descIdx],
                                    0, nullptr);
            uint32_t amountBits;
            std::memcpy(&amountBits, &sharpenAmount, sizeof(amountBits));
            const uint32_t rpc[4] = {outWidth, outHeight, amountBits, packedDst};
            vkCmdPushConstants(cb, rcasPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(rpc), rpc);
            vkCmdDispatch(cb, gx, gy, 1);
        }

        if (historyInvalidResolves_ > 0) --historyInvalidResolves_;
        historyValid_ = (historyInvalidResolves_ == 0);
    }

    void TaaResolve::rewritePostFinalizeDescriptors(const VkImageView* hdrOutPerFrame,
                                                    const VkImage* hdrOutImagePerFrame,
                                                    const VkImageView* swapchainViews,
                                                    const VkImage* swapchainImages) {
        if (!hdrOutPerFrame || !swapchainViews) return;
        if (hdrOutImagePerFrame) {
            postFinalizeSrcImage_.assign(hdrOutImagePerFrame, hdrOutImagePerFrame + framesInFlight_);
        }
        if (swapchainImages) {
            postFinalizeDstImage_.assign(swapchainImages, swapchainImages + imageCount_);
        }
        for (uint32_t f = 0; f < framesInFlight_; ++f) {
            for (uint32_t i = 0; i < imageCount_; ++i) {
                const uint32_t idx = f * imageCount_ + i;
                VkDescriptorImageInfo srcI{};
                srcI.sampler     = sampler_;
                // hdrOutPerFrame[f] is VK_NULL_HANDLE when PostComposite's
                // HDR-mode output scratch hasn't been allocated yet (no FSR/
                // DLSS upscaler active — the common case). Fall back to a TAA
                // history view so the descriptor never holds a genuinely
                // null image view; harmless, since recordPostFinalize (the
                // only reader of this set) is never called without an active
                // upscaler. NOT the swapchain view: this is a SAMPLED binding
                // and the swapchain lacks SAMPLED usage — even an unread
                // descriptor write must reference a usage-compatible view
                // (VUID-VkWriteDescriptorSet-descriptorType-00337).
                srcI.imageView   = hdrOutPerFrame[f] != VK_NULL_HANDLE
                        ? hdrOutPerFrame[f] : historyImagesPP_[0].view;
                srcI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorImageInfo dstI{};
                dstI.imageView   = swapchainViews[i];
                dstI.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet w[2]{};
                w[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[0].dstSet          = postFinalizeRcasSets_[idx];
                w[0].dstBinding      = 0;
                w[0].descriptorCount = 1;
                w[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w[0].pImageInfo      = &srcI;
                w[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[1].dstSet          = postFinalizeRcasSets_[idx];
                w[1].dstBinding      = 1;
                w[1].descriptorCount = 1;
                w[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[1].pImageInfo      = &dstI;
                vkUpdateDescriptorSets(ctx_.device(), 2, w, 0, nullptr);
            }
        }
    }

    void TaaResolve::recordPostFinalize(VkCommandBuffer cb, uint32_t frame, uint32_t imageIndex,
                                        uint32_t width, uint32_t height,
                                        bool sharpen, float sharpenAmount) {
        const uint32_t descIdx = frame * imageCount_ + imageIndex;
        if (sharpen) {
            // PostComposite's write (compute) → this RCAS read.
            VkMemoryBarrier2 mb{};
            mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo di{};
            di.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers    = &mb;
            vkCmdPipelineBarrier2(cb, &di);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rcasPipe_);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rcasPipeLayout_, 0, 1,
                                    &postFinalizeRcasSets_[descIdx], 0, nullptr);
            uint32_t amountBits;
            std::memcpy(&amountBits, &sharpenAmount, sizeof(amountBits));
            const uint32_t rpc[4] = {width, height, amountBits, 0u};
            vkCmdPushConstants(cb, rcasPipeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(rpc), rpc);
            vkCmdDispatch(cb, (width + 7u) / 8u, (height + 7u) / 8u, 1);
        } else {
            // Plain copy — both images are BGRA8 at the same (display)
            // extent, so a direct vkCmdCopyImage is exact and cheap (no
            // compute dispatch, no sampling/filtering).
            VkImageMemoryBarrier2 pre{};
            pre.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            pre.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            pre.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            pre.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            pre.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            pre.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            pre.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            pre.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre.image = postFinalizeSrcImage_[frame];
            pre.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            pre.subresourceRange.levelCount = 1;
            pre.subresourceRange.layerCount = 1;
            VkDependencyInfo dep{};
            dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &pre;
            vkCmdPipelineBarrier2(cb, &dep);

            VkImageCopy region{};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = 1;
            region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.dstSubresource.layerCount = 1;
            region.extent = {width, height, 1};
            vkCmdCopyImage(cb, postFinalizeSrcImage_[frame], VK_IMAGE_LAYOUT_GENERAL,
                          postFinalizeDstImage_[imageIndex], VK_IMAGE_LAYOUT_GENERAL,
                          1, &region);
        }
    }

}// namespace threepp::vulkan
