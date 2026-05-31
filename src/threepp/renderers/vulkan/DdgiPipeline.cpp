#include "threepp/renderers/vulkan/DdgiPipeline.hpp"

#include "threepp/renderers/vulkan/VulkanContext.hpp"

#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"
#include "threepp/renderers/vulkan/shaders/ddgi_update.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/ddgi_miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/ddgi_blend.comp.spv.h"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace threepp::vulkan {

    namespace {
        // Host mirror of ddgi_update.rgen's `DdgiUbo` (scalar block layout —
        // every member 4-byte aligned, tightly packed). Uploaded per frame once
        // the grid is sized to the scene AABB (a later increment); allocated
        // here so the buffer + binding exist.
        struct DdgiUboHost {
            float   gridOrigin[3];
            float   gridSpacing[3];
            int32_t probeCounts[3];
            int32_t raysPerProbe;
            int32_t totalProbes;
            float   rayRot0[3];
            float   rayRot1[3];
            float   rayRot2[3];
        };
    }// namespace

    DdgiPipeline::DdgiPipeline(VulkanContext& ctx, VkCommandPool cmdPool)
        : ctx_(ctx), cmdPool_(cmdPool) {
        computeGridDims();
        createImages();
        createBuffers();
        createUpdatePipeline();
        createSbt();
        createBlendPipeline();
        createDescriptors();
    }

    DdgiPipeline::~DdgiPipeline() {
        VkDevice d = ctx_.device();
        if (descPool_)       vkDestroyDescriptorPool(d, descPool_, nullptr);
        if (blendPipeline_)  vkDestroyPipeline(d, blendPipeline_, nullptr);
        if (blendLayout_)    vkDestroyPipelineLayout(d, blendLayout_, nullptr);
        if (blendDsLayout_)  vkDestroyDescriptorSetLayout(d, blendDsLayout_, nullptr);
        destroyBuffer(ctx_.allocator(), sbtBuf_);
        if (updatePipeline_) vkDestroyPipeline(d, updatePipeline_, nullptr);
        if (updateLayout_)   vkDestroyPipelineLayout(d, updateLayout_, nullptr);
        if (updateDsLayout_) vkDestroyDescriptorSetLayout(d, updateDsLayout_, nullptr);
        destroyBuffer(ctx_.allocator(), ddgiUbo_);
        destroyBuffer(ctx_.allocator(), rayRadiance_);
        for (auto& img : irradiance_) destroyImage2D(ctx_.allocator(), d, img);
    }

    void DdgiPipeline::computeGridDims() {
        probesX_ = kDdgiDefaultProbesX;
        probesY_ = kDdgiDefaultProbesY;
        probesZ_ = kDdgiDefaultProbesZ;
        totalProbes_  = probesX_ * probesY_ * probesZ_;
        raysPerProbe_ = kDdgiRaysPerProbe;
        irrRes_   = kDdgiIrradianceRes;
        border_   = kDdgiProbeBorder;
        tileSide_ = irrRes_ + 2 * border_;
        // Lay the per-probe tiles out in a roughly-square atlas.
        tilesPerRow_ = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(totalProbes_))));
        tileRows_    = (totalProbes_ + tilesPerRow_ - 1) / tilesPerRow_;
        atlasW_ = static_cast<uint32_t>(tilesPerRow_ * tileSide_);
        atlasH_ = static_cast<uint32_t>(tileRows_ * tileSide_);
    }

    void DdgiPipeline::transitionFreshImage(VkImage img) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = cmdPool_;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(ctx_.device(), &ai, &cb), "alloc one-shot cb(ddgi)");

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb(ddgi)");

        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        check(vkEndCommandBuffer(cb), "end one-shot cb(ddgi)");
        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cb;
        check(vkQueueSubmit(ctx_.graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot(ddgi)");
        check(vkQueueWaitIdle(ctx_.graphicsQueue()), "wait one-shot(ddgi)");
        vkFreeCommandBuffers(ctx_.device(), cmdPool_, 1, &cb);
    }

    void DdgiPipeline::createImages() {
        for (auto& img : irradiance_) {
            img = Image2D{};
            img.width  = atlasW_;
            img.height = atlasH_;
            img.format = VK_FORMAT_R16G16B16A16_SFLOAT;

            VkImageCreateInfo ici{};
            ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = img.format;
            ici.extent        = {atlasW_, atlasH_, 1};
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            check(vmaCreateImage(ctx_.allocator(), &ici, &aci, &img.image, &img.alloc, nullptr),
                  "vmaCreateImage(ddgi.irradiance)");

            transitionFreshImage(img.image);

            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = img.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = img.format;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            check(vkCreateImageView(ctx_.device(), &vci, nullptr, &img.view),
                  "vkCreateImageView(ddgi.irradiance)");
        }
    }

    void DdgiPipeline::createBuffers() {
        rayRadiance_ = createBuffer(
                ctx_.allocator(), ctx_.device(),
                static_cast<VkDeviceSize>(totalProbes_) * raysPerProbe_ * sizeof(float) * 4,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

        ddgiUbo_ = createBuffer(
                ctx_.allocator(), ctx_.device(), sizeof(DdgiUboHost),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    void DdgiPipeline::createUpdatePipeline() {
        // Own descriptor set layout: 0=TLAS, 1=DDGI UBO, 2=ray-radiance (write),
        // 3=env sampler (read in the miss shader).
        std::array<VkDescriptorSetLayoutBinding, 4> b{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[3].descriptorCount = 1; b[3].stageFlags = VK_SHADER_STAGE_MISS_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &updateDsLayout_),
              "vkCreateDescriptorSetLayout(ddgi.update)");

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts    = &updateDsLayout_;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &updateLayout_),
              "vkCreatePipelineLayout(ddgi.update)");

        auto loadModule = [this](const uint32_t* code, size_t size) {
            VkShaderModuleCreateInfo smci{};
            smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = size;
            smci.pCode    = code;
            VkShaderModule m = VK_NULL_HANDLE;
            check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &m), "vkCreateShaderModule(ddgi)");
            return m;
        };
        VkShaderModule rgenMod = loadModule(kDdgiUpdateRgenSpv, sizeof(kDdgiUpdateRgenSpv));
        VkShaderModule missMod = loadModule(kDdgiMissRmissSpv,  sizeof(kDdgiMissRmissSpv));

        std::array<VkPipelineShaderStageCreateInfo, 2> stg{};
        for (auto& s : stg) { s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; s.pName = "main"; }
        stg[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; stg[0].module = rgenMod;
        stg[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;   stg[1].module = missMod;

        // Two general groups (rgen, miss); no hit group — the update rgen traces
        // with gl_RayFlagsSkipClosestHitShaderEXT, so the hit SBT region is empty.
        std::array<VkRayTracingShaderGroupCreateInfoKHR, 2> grp{};
        for (auto& g : grp) {
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.generalShader = g.closestHitShader = g.anyHitShader = g.intersectionShader = VK_SHADER_UNUSED_KHR;
        }
        grp[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; grp[0].generalShader = 0;
        grp[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR; grp[1].generalShader = 1;

        VkRayTracingPipelineCreateInfoKHR rci{};
        rci.sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rci.stageCount = static_cast<uint32_t>(stg.size());
        rci.pStages    = stg.data();
        rci.groupCount = static_cast<uint32_t>(grp.size());
        rci.pGroups    = grp.data();
        rci.maxPipelineRayRecursionDepth = 1;
        rci.layout     = updateLayout_;
        check(ctx_.rt().createRayTracingPipelines(
                      ctx_.device(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rci, nullptr, &updatePipeline_),
              "vkCreateRayTracingPipelinesKHR(ddgi.update)");

        vkDestroyShaderModule(ctx_.device(), rgenMod, nullptr);
        vkDestroyShaderModule(ctx_.device(), missMod, nullptr);
    }

    void DdgiPipeline::createSbt() {
        const auto& props = ctx_.rtPipelineProperties();
        const uint32_t handleSize        = props.shaderGroupHandleSize;
        const uint32_t handleAlignment   = props.shaderGroupHandleAlignment;
        const uint32_t baseAlignment     = props.shaderGroupBaseAlignment;
        const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

        constexpr uint32_t groupCount = 2;// rgen, miss
        const uint32_t handlesSize = groupCount * handleSize;
        std::vector<uint8_t> handles(handlesSize);
        check(ctx_.rt().getRayTracingShaderGroupHandles(
                      ctx_.device(), updatePipeline_, 0, groupCount, handlesSize, handles.data()),
              "vkGetRayTracingShaderGroupHandlesKHR(ddgi)");

        const uint32_t rgenBytes = alignUp(handleSizeAligned, baseAlignment);
        const uint32_t missBytes = alignUp(handleSizeAligned, baseAlignment);
        const VkDeviceSize sbtSize = static_cast<VkDeviceSize>(rgenBytes) + missBytes;

        sbtBuf_ = createBuffer(
                ctx_.allocator(), ctx_.device(), sbtSize,
                VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

        void* mapped = nullptr;
        vmaMapMemory(ctx_.allocator(), sbtBuf_.alloc, &mapped);
        std::memset(mapped, 0, sbtSize);
        uint8_t* dst = static_cast<uint8_t*>(mapped);
        std::memcpy(dst,            handles.data() + 0 * handleSize, handleSize);
        std::memcpy(dst + rgenBytes, handles.data() + 1 * handleSize, handleSize);
        vmaUnmapMemory(ctx_.allocator(), sbtBuf_.alloc);

        const VkDeviceAddress base = sbtBuf_.address;
        rgenRgn_.deviceAddress = base;
        rgenRgn_.stride        = rgenBytes;
        rgenRgn_.size          = rgenBytes;
        missRgn_.deviceAddress = base + rgenBytes;
        missRgn_.stride        = handleSizeAligned;
        missRgn_.size          = missBytes;
        hitRgn_  = {};// no hit shaders (SkipClosestHit)
        callRgn_ = {};
    }

    void DdgiPipeline::createBlendPipeline() {
        std::array<VkDescriptorSetLayoutBinding, 3> b{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dlci{};
        dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dlci.bindingCount = static_cast<uint32_t>(b.size());
        dlci.pBindings    = b.data();
        check(vkCreateDescriptorSetLayout(ctx_.device(), &dlci, nullptr, &blendDsLayout_),
              "vkCreateDescriptorSetLayout(ddgi.blend)");

        // ddgi_blend.comp push constants: ivec4 + ivec4 + vec4 + 3×vec4 = 96 B.
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = 96;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &blendDsLayout_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcRange;
        check(vkCreatePipelineLayout(ctx_.device(), &plci, nullptr, &blendLayout_),
              "vkCreatePipelineLayout(ddgi.blend)");

        VkShaderModuleCreateInfo smci{};
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(kDdgiBlendCompSpv);
        smci.pCode    = kDdgiBlendCompSpv;
        VkShaderModule mod = VK_NULL_HANDLE;
        check(vkCreateShaderModule(ctx_.device(), &smci, nullptr, &mod),
              "vkCreateShaderModule(ddgi.blend)");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName  = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage  = stage;
        cpci.layout = blendLayout_;
        check(vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &blendPipeline_),
              "vkCreateComputePipelines(ddgi.blend)");
        vkDestroyShaderModule(ctx_.device(), mod, nullptr);
    }

    void DdgiPipeline::createDescriptors() {
        std::array<VkDescriptorPoolSize, 4> sizes{};
        sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; sizes[0].descriptorCount = 1;
        sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;             sizes[1].descriptorCount = 1;
        sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;             sizes[2].descriptorCount = 1 + kPingPong;
        sizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;              sizes[3].descriptorCount = 2 * kPingPong;
        // (one COMBINED_IMAGE_SAMPLER for the update set's env binding)
        std::array<VkDescriptorPoolSize, 5> allSizes{};
        for (size_t i = 0; i < sizes.size(); ++i) allSizes[i] = sizes[i];
        allSizes[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; allSizes[4].descriptorCount = 1;

        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets       = 1 + kPingPong;// update set + 2 blend sets
        dpci.poolSizeCount = static_cast<uint32_t>(allSizes.size());
        dpci.pPoolSizes    = allSizes.data();
        check(vkCreateDescriptorPool(ctx_.device(), &dpci, nullptr, &descPool_),
              "vkCreateDescriptorPool(ddgi)");

        VkDescriptorSetAllocateInfo uai{};
        uai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        uai.descriptorPool     = descPool_;
        uai.descriptorSetCount = 1;
        uai.pSetLayouts        = &updateDsLayout_;
        check(vkAllocateDescriptorSets(ctx_.device(), &uai, &updateSet_),
              "vkAllocateDescriptorSets(ddgi.update)");

        for (auto& set : blendSets_) {
            VkDescriptorSetAllocateInfo bai{};
            bai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            bai.descriptorPool     = descPool_;
            bai.descriptorSetCount = 1;
            bai.pSetLayouts        = &blendDsLayout_;
            check(vkAllocateDescriptorSets(ctx_.device(), &bai, &set),
                  "vkAllocateDescriptorSets(ddgi.blend)");
        }
        // Descriptor *contents* (TLAS / env / atlas / ray buffer) are wired when
        // DDGI is enabled — until then the sets are allocated but unused.
    }

}// namespace threepp::vulkan
